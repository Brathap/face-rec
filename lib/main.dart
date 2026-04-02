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
      title: 'Face Gatekeeper',
      theme: ThemeData.dark(),
      home: OfflineFaceGatekeeper(cameras: cameras),
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
  String _statusMessage = "Select an action below";
  
  List<double>? _registeredFaceEmbedding;

  @override
  void initState() {
    super.initState();
    _loadRegisteredFace();
    _initializeServices();
  }

  // Load the saved face from the phone's offline memory
  Future<void> _loadRegisteredFace() async {
    final prefs = await SharedPreferences.getInstance();
    final savedFace = prefs.getString('my_face_embedding');
    if (savedFace != null) {
      _registeredFaceEmbedding = savedFace.split(',').map((e) => double.parse(e)).toList();
    }
  }

  // Save the face to the phone's offline memory
  Future<void> _saveRegisteredFace(List<double> embedding) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('my_face_embedding', embedding.join(','));
    _registeredFaceEmbedding = embedding;
    if (mounted) {
      setState(() => _statusMessage = "✅ Face Registered Successfully!");
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
      ResolutionPreset.low,
      enableAudio: false,
      imageFormatGroup: Platform.isAndroid ? ImageFormatGroup.yuv420 : ImageFormatGroup.bgra8888,
    );

    await _cameraController.initialize();
    
    _faceDetector = FaceDetector(
      options: FaceDetectorOptions(performanceMode: FaceDetectorMode.fast),
    );
    
    try {
      _interpreter = await Interpreter.fromAsset('assets/mobilefacenet.tflite');
    } catch (e) {
      print("Error loading model: $e");
    }

    if (mounted) {
      setState(() => _isInitialized = true);
    }
  }

  Future<void> _scanFace({required bool isRegistering}) async {
    if (_isProcessing) return;
    
    setState(() {
      _isProcessing = true;
      _statusMessage = "Scanning face...";
    });

    try {
      // 1. Take a picture
      final image = await _cameraController.takePicture();
      final inputImage = InputImage.fromFilePath(image.path);
      
      // 2. Find the face in the picture
      final faces = await _faceDetector.processImage(inputImage);

      if (faces.isEmpty) {
        if (mounted) setState(() => _statusMessage = "❌ No face detected. Try again.");
        _isProcessing = false;
        return;
      }

      // 3. Extract the math from the face
      final Face face = faces.first;
      List<double> currentFaceEmbedding = await _getFaceEmbedding(image.path, face.boundingBox);

      // 4. Save it or Compare it
      if (isRegistering) {
        await _saveRegisteredFace(currentFaceEmbedding);
      } else {
        _verifyFace(currentFaceEmbedding);
      }

    } catch (e) {
      if (mounted) setState(() => _statusMessage = "Error: Something went wrong.");
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  void _verifyFace(List<double> currentFace) {
    if (_registeredFaceEmbedding == null) {
      setState(() => _statusMessage = "❌ No face registered yet. Please register first.");
      return;
    }

    // Euclidean Distance formula to compare the two faces
    double distance = 0.0;
    for (int i = 0; i < currentFace.length; i++) {
      distance += pow((currentFace[i] - _registeredFaceEmbedding![i]), 2);
    }
    distance = sqrt(distance);

    // 1.0 is the standard threshold. Lower is stricter.
    if (distance < 1.0) {
      setState(() => _statusMessage = "✅ Identity Confirmed! Access Granted.");
    } else {
      setState(() => _statusMessage = "🚫 Access Denied. Face does not match.");
    }
  }

  // --- SAFE MATH TRANSLATION ---
  Future<List<double>> _getFaceEmbedding(String imagePath, Rect boundingBox) async {
    final bytes = await File(imagePath).readAsBytes();
    img.Image? originalImage = img.decodeImage(bytes);
    if (originalImage == null) return [];

    // Crop to just the face
    img.Image croppedFace = img.copyCrop(
      originalImage,
      x: boundingBox.left.toInt(),
      y: boundingBox.top.toInt(),
      width: boundingBox.width.toInt(),
      height: boundingBox.height.toInt(),
    );

    // Resize to the 112x112 requirement of the TFLite model
    img.Image resizedFace = img.copyResize(croppedFace, width: 112, height: 112);
    
    // Convert to a 4D nested list [1][112][112][3] to prevent library memory bugs
    List input = _imageToNestedList(resizedFace, 112, 127.5, 127.5);
    
    // Create an empty output list [1][192]
    var output = List.generate(1, (i) => List.filled(192, 0.0)); 
    
    _interpreter.run(input, output);
    return output[0];
  }

  // Pure Dart conversion to avoid Float32List reshape bugs
  List<List<List<List<double>>>> _imageToNestedList(img.Image image, int inputSize, double mean, double std) {
    return List.generate(1, (i) => 
      List.generate(inputSize, (y) => 
        List.generate(inputSize, (x) {
          var pixel = image.getPixel(x, y);
          return [
            (pixel.r - mean) / std,
            (pixel.g - mean) / std,
            (pixel.b - mean) / std
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

    return Scaffold(
      backgroundColor: Colors.black,
      body: Column(
        children: [
          Expanded(
            child: Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(_cameraController),
                Center(
                  child: Container(
                    width: 250, height: 300,
                    decoration: BoxDecoration(
                      border: Border.all(
                        color: _statusMessage.contains("✅") ? Colors.green : Colors.blueAccent, 
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
                  style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white),
                ),
                const SizedBox(height: 24),
                Row(
                  children: [
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: _isProcessing ? null : () => _scanFace(isRegistering: true),
                        icon: const Icon(Icons.face),
                        label: const Text("1. Register"),
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(vertical: 16),
                          backgroundColor: Colors.blueGrey,
                        ),
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: ElevatedButton.icon(
                        onPressed: _isProcessing ? null : () => _scanFace(isRegistering: false),
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
