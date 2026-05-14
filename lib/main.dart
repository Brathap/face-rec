import 'dart:io';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;
import 'package:shared_preferences/shared_preferences.dart';

// --- APP STARTUP ---
List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    cameras = await availableCameras();
  } catch (e) {
    print('Error initializing cameras: $e');
  }
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Face Gatekeeper Pro',
      theme: ThemeData.dark(),
      home: OfflineFaceGatekeeper(cameras: cameras),
      debugShowCheckedModeBanner: false,
    );
  }
}

// --- GATEKEEPER LOGIC ---
class OfflineFaceGatekeeper extends StatefulWidget {
  final List<CameraDescription> cameras;

  const OfflineFaceGatekeeper({Key? key, required this.cameras}) : super(key: key);

  @override
  _OfflineFaceGatekeeperState createState() => _OfflineFaceGatekeeperState();
}

class _OfflineFaceGatekeeperState extends State<OfflineFaceGatekeeper> {
  late CameraController _cameraController;
  late FaceDetector _faceDetector;
  late Interpreter _interpreter;
  
  bool _isProcessing = false;
  bool _isInitialized = false;
  String _statusMessage = "Ready. Please select an action.";
  
  List<List<double>>? _registeredFaceEmbeddings;
  
  // Registration & Liveness Variables
  int _registrationStep = 0;
  final List<List<double>> _tempEmbeddings = [];
  
  int _challengeType = -1; // 0: Smile, 1: Look Left, 2: Look Right
  int _livenessAttempts = 0;

  @override
  void initState() {
    super.initState();
    _loadRegisteredFace();
    _initializeServices();
  }

  Future<void> _loadRegisteredFace() async {
    final prefs = await SharedPreferences.getInstance();
    final savedFaces = prefs.getStringList('my_face_embeddings');
    if (savedFaces != null && savedFaces.isNotEmpty) {
      _registeredFaceEmbeddings = savedFaces.map((faceString) {
        return faceString.split(',').map((e) => double.parse(e)).toList();
      }).toList();
    }
  }

  Future<void> _saveRegisteredFace(List<List<double>> embeddings) async {
    final prefs = await SharedPreferences.getInstance();
    List<String> stringifiedFaces = embeddings.map((e) => e.join(',')).toList();
    await prefs.setStringList('my_face_embeddings', stringifiedFaces);
    
    _registeredFaceEmbeddings = embeddings;
    if (mounted) {
      setState(() {
        _statusMessage = "✅ 3 Lighting Conditions Registered!";
        _registrationStep = 0;
        _tempEmbeddings.clear();
      });
    }
  }

  Future<void> _initializeServices() async {
    if (widget.cameras.isEmpty) {
      setState(() => _statusMessage = "No cameras found.");
      return;
    }

    final frontCamera = widget.cameras.firstWhere(
      (camera) => camera.lensDirection == CameraLensDirection.front,
      orElse: () => widget.cameras.first,
    );
    
    _cameraController = CameraController(
      frontCamera,
      ResolutionPreset.high, 
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid ? ImageFormatGroup.yuv420 : ImageFormatGroup.bgra8888,
    );

    await _cameraController.initialize();
    await _cameraController.setExposureMode(ExposureMode.auto);
    await _cameraController.setFocusMode(FocusMode.auto); 
    
    _faceDetector = FaceDetector(
      options: FaceDetectorOptions(
        performanceMode: FaceDetectorMode.accurate,
        enableClassification: true, // Needed for Smile challenge
        enableLandmarks: true,
        enableContours: true, 
      ),
    );
    
    try {
      _interpreter = await Interpreter.fromAsset('assets/mobilefacenet.tflite');
    } catch (e) {
      print("Error loading model: $e");
    }

    if (mounted) setState(() => _isInitialized = true);
  }

  // ==========================================
  // ACTIVE LIVENESS CHALLENGE (RANDOMIZED)
  // ==========================================
  Future<void> _startUnlockSequence() async {
    if (_registeredFaceEmbeddings == null || _registeredFaceEmbeddings!.isEmpty) {
      setState(() => _statusMessage = "❌ Please Register first.");
      return;
    }

    setState(() {
      _isProcessing = true;
      _livenessAttempts = 0;
      // Randomly pick a challenge: 0 (Smile), 1 (Left), 2 (Right)
      _challengeType = Random().nextInt(3); 
      _statusMessage = _getChallengeText(_challengeType);
    });
    
    _runLivenessLoop();
  }

  String _getChallengeText(int type) {
    if (type == 0) return "🤖 HUMAN TEST: Please SMILE 😁";
    if (type == 1) return "🤖 HUMAN TEST: Turn head LEFT 👈";
    return "🤖 HUMAN TEST: Turn head RIGHT 👉";
  }

  Future<void> _runLivenessLoop() async {
    if (!mounted || !_isProcessing) return;

    _livenessAttempts++;
    if (_livenessAttempts > 20) { // Timeout after ~8 seconds
      if (mounted) setState(() {
        _statusMessage = "⏱️ Liveness Timeout. Proxy suspected.";
        _isProcessing = false;
      });
      return;
    }

    try {
      final image = await _cameraController.takePicture();
      final inputImage = InputImage.fromFilePath(image.path);
      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isNotEmpty) {
        final face = faces.first;
        bool challengePassed = false;

        // Verify the dynamic action
        if (_challengeType == 0 && face.smilingProbability != null && face.smilingProbability! > 0.75) {
          challengePassed = true;
        } else if (_challengeType == 1 && face.headEulerAngleY != null && face.headEulerAngleY! < -20) {
          challengePassed = true; // Looking Left
        } else if (_challengeType == 2 && face.headEulerAngleY != null && face.headEulerAngleY! > 20) {
          challengePassed = true; // Looking Right
        }

        if (challengePassed) {
          setState(() => _statusMessage = "✅ Human Verified! Look straight to unlock...");
          // Give them 1.5 seconds to face the camera normally again
          await Future.delayed(const Duration(milliseconds: 1500));
          // Proceed to actual facial recognition
          _scanFace(isRegistering: false); 
          return; // Exit the loop entirely
        }
      }
    } catch (e) {
      print("Liveness error: $e");
    }

    // If challenge not met, wait 400ms and check again
    if (_isProcessing) {
      await Future.delayed(const Duration(milliseconds: 400));
      _runLivenessLoop();
    }
  }

  // ==========================================
  // CORE SCANNER & GEOMETRY BLOCKERS
  // ==========================================
  Future<void> _scanFace({required bool isRegistering}) async {
    setState(() {
      _isProcessing = true;
      if (isRegistering) _statusMessage = "Registration Step ${_registrationStep + 1}/3...";
    });

    try {
      final image = await _cameraController.takePicture();
      final inputImage = InputImage.fromFilePath(image.path);
      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isEmpty) {
        if (mounted) setState(() => _statusMessage = "❌ No face detected.");
        _isProcessing = false;
        return;
      }

      final Face face = faces.first;

      // 1. STRICT ASPECT RATIO (Blocks partial printed photos)
      double boundingBoxAspectRatio = face.boundingBox.height / face.boundingBox.width;
      if (boundingBoxAspectRatio < 1.15 || boundingBoxAspectRatio > 1.65) {
        if (mounted) setState(() => _statusMessage = "🚫 Show your ENTIRE face. Partial scans blocked.");
        _isProcessing = false;
        return; 
      }

      // 2. CONTOUR VALIDATION (Ensures physical jawline is in the camera)
      final faceOval = face.contours[FaceContourType.face];
      if (faceOval == null || faceOval.points.length < 30) {
        if (mounted) setState(() => _statusMessage = "🚫 Keep your whole head in the frame.");
        _isProcessing = false;
        return; 
      }

      // 3. POSE ESTIMATION (Must be perfectly straight for accurate math)
      if (face.headEulerAngleY!.abs() > 10 || face.headEulerAngleZ!.abs() > 10) {
        if (mounted) setState(() => _statusMessage = "❌ Look dead straight at the camera.");
        _isProcessing = false;
        return;
      }

      // 4. SCREEN DISTANCE (Face must be a healthy size)
      double screenWidth = MediaQuery.of(context).size.width;
      if (face.boundingBox.width < (screenWidth * 0.35)) {
         if (mounted) setState(() => _statusMessage = "❌ Move closer to the camera.");
         _isProcessing = false;
         return;
      }

      // Extract math from the verified, full, perfect face
      List<double> currentFaceEmbedding = await _getFaceEmbedding(image.path, face.boundingBox);

      if (isRegistering) {
        await _handleRegistration(currentFaceEmbedding);
      } else {
        _verifyFace(currentFaceEmbedding);
      }

    } catch (e) {
      if (mounted) setState(() => _statusMessage = "Error: Something went wrong.");
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  Future<void> _handleRegistration(List<double> embedding) async {
    _tempEmbeddings.add(embedding);
    _registrationStep++;

    if (_registrationStep < 3) {
      setState(() => _statusMessage = "✅ Scan $_registrationStep complete. Change lighting/shadows and scan again.");
      _isProcessing = false;
    } else {
      await _saveRegisteredFace(List.from(_tempEmbeddings));
      _isProcessing = false;
    }
  }

  void _verifyFace(List<double> currentFace) {
    double highestSimilarity = 0.0;

    for (var savedTemplate in _registeredFaceEmbeddings!) {
      double dotProduct = 0.0;
      double normA = 0.0;
      double normB = 0.0;
      
      for (int i = 0; i < currentFace.length; i++) {
        dotProduct += currentFace[i] * savedTemplate[i];
        normA += pow(currentFace[i], 2);
        normB += pow(savedTemplate[i], 2);
      }
      
      double similarity = dotProduct / (sqrt(normA) * sqrt(normB));
      if (similarity > highestSimilarity) {
        highestSimilarity = similarity;
      }
    }

    if (highestSimilarity > 0.82) { 
      setState(() => _statusMessage = "✅ Attendance Marked! (Sim: ${(highestSimilarity*100).toStringAsFixed(1)}%)");
      // TODO: Push attendance to Supabase 
    } else {
      setState(() => _statusMessage = "🚫 Identity Denied. Proxy Detected. (Sim: ${(highestSimilarity*100).toStringAsFixed(1)}%)");
    }
  }

  Future<List<double>> _getFaceEmbedding(String imagePath, Rect boundingBox) async {
    final bytes = await File(imagePath).readAsBytes();
    img.Image? originalImage = img.decodeImage(bytes);
    if (originalImage == null) return [];

    int x = max(0, boundingBox.left.toInt());
    int y = max(0, boundingBox.top.toInt());
    int width = min(originalImage.width - x, boundingBox.width.toInt());
    int height = min(originalImage.height - y, boundingBox.height.toInt());

    img.Image croppedFace = img.copyCrop(originalImage, x: x, y: y, width: width, height: height);
    img.Image resizedFace = img.copyResize(croppedFace, width: 112, height: 112);
    
    List input = _imageToNestedList(resizedFace, 112);
    var output = List.generate(1, (i) => List.filled(192, 0.0)); 
    
    _interpreter.run(input, output);
    return output[0];
  }

  List<List<List<List<double>>>> _imageToNestedList(img.Image image, int inputSize) {
    return List.generate(1, (i) => 
      List.generate(inputSize, (y) => 
        List.generate(inputSize, (x) {
          var pixel = image.getPixel(x, y);
          return [
            (pixel.r - 127.5) / 127.5,
            (pixel.g - 127.5) / 127.5,
            (pixel.b - 127.5) / 127.5
          ];
        })
      )
    );
  }

  @override
  void dispose() {
    _cameraController.dispose();
    _faceDetector.close();
    _interpreter.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isInitialized) {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    final size = MediaQuery.of(context).size;
    var scale = size.aspectRatio * _cameraController.value.aspectRatio;
    if (scale < 1) scale = 1 / scale;

    return Scaffold(
      backgroundColor: Colors.black,
      body: Column(
        children: [
          Expanded(
            child: Stack(
              fit: StackFit.expand,
              children: [
                Transform.scale(
                  scale: scale,
                  child: Center(
                    child: CameraPreview(_cameraController),
                  ),
                ),
                Center(
                  child: Container(
                    width: 250, height: 350,
                    decoration: BoxDecoration(
                      border: Border.all(
                        color: _statusMessage.contains("✅") 
                            ? Colors.green 
                            : _statusMessage.contains("❌") || _statusMessage.contains("🚫")
                                ? Colors.red
                                : Colors.blueAccent, 
                        width: 4
                      ),
                      borderRadius: BorderRadius.circular(150),
                    ),
                  ),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.all(24),
            decoration: const BoxDecoration(
              color: Color(0xFF1E1E1E),
              borderRadius: BorderRadius.only(topLeft: Radius.circular(20), topRight: Radius.circular(20)),
            ),
            child: Column(
              children: [
                Text(
                  _statusMessage, 
                  textAlign: TextAlign.center,
                  style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white),
                ),
                const SizedBox(height: 24),
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton.icon(
                        // Disables button if already processing
                        onPressed: _isProcessing ? null : () => _scanFace(isRegistering: true),
                        icon: const Icon(Icons.face),
                        label: Text(_registrationStep > 0 ? "Scan ${_registrationStep + 1}" : "1. Register"),
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          backgroundColor: Colors.blueGrey,
                        ),
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: ElevatedButton.icon(
                        // Triggers the new Liveness Challenge sequence
                        onPressed: _isProcessing ? null : () => _startUnlockSequence(),
                        icon: const Icon(Icons.lock_open),
                        label: const Text("2. Unlock"),
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          backgroundColor: Colors.green,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 20),
              ],
            ),
          )
        ],
      ),
    );
  }
}
