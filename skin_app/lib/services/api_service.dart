// api_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

class ApiService {
  // iPhone için kendi IP adresini kullanıyorsun, bu kalsın:
  final String _apiUrl = "http://192.168.1.4:8000/api/scan-image";

  Future<List<dynamic>> scanImage(XFile imageFile) async {
    print("API servisi çağrıldı: $_apiUrl");
    final uri = Uri.parse(_apiUrl);

    final request = http.MultipartRequest('POST', uri);
    request.files.add(
      await http.MultipartFile.fromPath(
        'image_file',
        imageFile.path,
      ),
    );

    print("Fotoğraf sunucuya yükleniyor...");

    try {
      final streamedResponse = await request.send();
      final response = await http.Response.fromStream(streamedResponse);

      print("HTTP status: ${response.statusCode}");

      if (response.statusCode == 200) {
        print("Analiz başarılı! JSON alınıyor...");

        final String responseBody = utf8.decode(response.bodyBytes);
        final decoded = jsonDecode(responseBody);

        // decoded: Map<String, dynamic> bekliyoruz
        if (decoded is Map<String, dynamic>) {
          final analysis = decoded['analysis_result'];

          if (analysis is List) {
            return analysis;
          } else {
            print("analysis_result list değil: $analysis");
            return [];
          }
        } else if (decoded is List) {
          // Backend'i değiştirirsen, bu da çalışsın diye fallback
          return decoded;
        } else {
          print("Beklenmeyen JSON tipi: ${decoded.runtimeType}");
          return [];
        }
      } else {
        print("Sunucu hatası: ${response.statusCode}");
        print("Sunucu mesajı: ${response.body}");
        return [];
      }
    } catch (e) {
      print("Ağ/Bağlantı hatası: $e");
      return [];
    }
  }
}
