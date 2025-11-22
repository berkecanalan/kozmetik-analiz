// lib/screens/user_info_screen.dart

import 'package:flutter/material.dart';
// Gerekli paketleri import ediyoruz:
import 'package:image_picker/image_picker.dart';
// Proje adınız 'skin_app' ise bu yol DOĞRU OLMALI
import 'package:skin_app/services/api_service.dart'; 

class UserInfoScreen extends StatefulWidget {
  const UserInfoScreen({super.key});

  @override
  State<UserInfoScreen> createState() => _UserInfoScreenState();
}

class _UserInfoScreenState extends State<UserInfoScreen> {
  
  // Form kontrolcüleri (sizde zaten var)
  final _formKey = GlobalKey<FormState>();
  final TextEditingController _nameController = TextEditingController();
  final TextEditingController _heightController = TextEditingController();
  final TextEditingController _weightController = TextEditingController();
  String? _selectedSkinType;
  final List<String> _skinTypes = ['Yağlı', 'Kuru', 'Karma', 'Normal'];

  // --- EKSİK OLAN DEĞİŞKENLER (HATA GİDERİCİ) ---
  // Bu değişkenler _takePictureAndScan fonksiyonunun çalışması için gereklidir.
  // Bunları sınıfınızın en üstüne (State sınıfının içine) ekleyin.
  final ImagePicker _picker = ImagePicker();
  final ApiService _apiService = ApiService(); // Hata veren yerlerden biri buydu
  bool _isLoading = false;
  List<dynamic> _analysisResults = []; // Hata veren yerlerden biri buydu
  
  @override
  void dispose() {
    _nameController.dispose();
    _heightController.dispose();
    _weightController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Cilt Analiz Uygulaması'),
        backgroundColor: Colors.blue.shade50,
      ),
      body: Stack( // Yüklenme animasyonu için Stack
        children: [
          SingleChildScrollView(
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Form(
                key: _formKey,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    // İsim Alanı
                    TextFormField(
                      controller: _nameController,
                      decoration: const InputDecoration(
                        labelText: 'İsim Soyisim',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.person_outline),
                      ),
                      validator: (value) {
                        if (value == null || value.isEmpty) return 'Lütfen isminizi girin';
                        return null;
                      },
                    ),
                    const SizedBox(height: 16.0),

                    // Boy Alanı
                    TextFormField(
                      controller: _heightController,
                      decoration: const InputDecoration(
                        labelText: 'Boy (cm)',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.height),
                      ),
                      keyboardType: TextInputType.number,
                      validator: (value) {
                        if (value == null || value.isEmpty) return 'Lütfen boyunuzu girin';
                        if (double.tryParse(value) == null) return 'Geçerli bir sayı girin';
                        return null;
                      },
                    ),
                    const SizedBox(height: 16.0),

                    // Kilo Alanı
                    TextFormField(
                      controller: _weightController,
                      decoration: const InputDecoration(
                        labelText: 'Kilo (kg)',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.monitor_weight),
                      ),
                      keyboardType: TextInputType.number,
                      validator: (value) {
                        if (value == null || value.isEmpty) return 'Lütfen kilonuzu girin';
                        if (double.tryParse(value) == null) return 'Geçerli bir sayı girin';
                        return null;
                      },
                    ),
                    const SizedBox(height: 16.0),

                    // Cilt Tipi (Açılır Menü)
                    DropdownButtonFormField<String>(
                      initialValue: _selectedSkinType,
                      decoration: const InputDecoration(
                        labelText: 'Cilt Tipi',
                        border: OutlineInputBorder(),
                        prefixIcon: Icon(Icons.face),
                      ),
                      hint: const Text('Seçiniz...'),
                      items: _skinTypes.map((String type) {
                        return DropdownMenuItem<String>(
                          value: type,
                          child: Text(type),
                        );
                      }).toList(),
                      onChanged: (String? newValue) {
                        setState(() {
                          _selectedSkinType = newValue;
                        });
                      },
                      validator: (value) {
                        if (value == null) return 'Lütfen cilt tipinizi seçin';
                        return null;
                      },
                    ),
                    
                    const SizedBox(height: 100), // Buton için boşluk
                  ],
                ),
              ),
            ),
          ),

          // Yüklenme animasyonu
          if (_isLoading)
            Container(
              color: Colors.black.withOpacity(0.5),
              child: const Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(color: Colors.white),
                    SizedBox(height: 20),
                    Text(
                      "Analiz ediliyor...\nBu işlem 1 dakikaya kadar sürebilir.",
                      textAlign: TextAlign.center,
                      style: TextStyle(color: Colors.white, fontSize: 16),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),

      // Fotoğraf Çekme Butonu
      floatingActionButton: FloatingActionButton.large(
        // Artık _submitForm değil, _submitAndScan'i çağırıyoruz
        onPressed: _isLoading ? null : _submitAndScan, 
        tooltip: 'Ürün Fotoğrafı Çek',
        backgroundColor: _isLoading ? Colors.grey : Colors.blue,
        child: const Icon(Icons.camera_alt_rounded),
      ),
      floatingActionButtonLocation: FloatingActionButtonLocation.centerFloat,
    );
  }

  // --- GÜNCELLENMİŞ FONKSİYON: _submitForm DEĞİL, _submitAndScan ---
  // (Önceki 'ScannerScreen'i çağıran _submitForm fonksiyonunuzu bununla değiştirin)
  void _submitAndScan() async {
    // 1. Formu doğrula
    if (!_formKey.currentState!.validate()) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text('Lütfen tüm alanları doğru şekilde doldurun.'),
          backgroundColor: Colors.red,
        ),
      );
      return; 
    }

    // 2. Form geçerliyse, verileri al
    String name = _nameController.text;
    String height = _heightController.text;
    String weight = _weightController.text;
    String skinType = _selectedSkinType!;
    print('--- Kullanıcı Bilgileri Kaydedildi ---');
    print('İsim: $name, Boy: $height, Kilo: $weight, Cilt: $skinType');
    print('------------------------------------');

    // 3. Fotoğrafı Çek (Kamera ve Kalite ayarı ile)
    print("Kamera açılıyor (Yüksek Kalite Modu)...");
    final XFile? image = await _picker.pickImage(
      source: ImageSource.camera,
      imageQuality: 90, // <-- ÖNEMLİ DEĞİŞİKLİK BURADA
    );

    if (image == null) {
      print("Kullanıcı fotoğraf çekmeyi iptal etti.");
      return;
    }

    setState(() {
      _isLoading = true;
    });

    // 4. API Servisini Çağır
    print("Fotoğraf çekildi, API'ye gönderiliyor...");
    final results = await _apiService.scanImage(image);

    // 5. Yüklenmeyi durdur ve sonuçları al
    setState(() {
      _isLoading = false;
      _analysisResults = results; // Hata veren satırlardan biri buydu
    });

    print("Alınan JSON sonucu: $_analysisResults"); // Hata veren satırlardan biri buydu

    // 6. Sonuçları göster
    if (mounted) {
      _showResultsModal(_analysisResults); // Hata veren satırlardan biri buydu
    }
  }

  // --- EKSİK OLAN FONKSİYON (HATA GİDERİCİ) ---
  // _showResultsModal fonksiyonunu _UserInfoScreenState sınıfının içine ekleyin.
  void _showResultsModal(List<dynamic> results) {
  showModalBottomSheet(
    context: context,
    builder: (context) {
      return Container(
        height: MediaQuery.of(context).size.height * 0.5,
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Analiz Sonuçları (${results.length} bileşen bulundu)',
              style: Theme.of(context).textTheme.headlineSmall,
            ),
            const Divider(height: 20),

            Expanded(
              child: results.isEmpty
                  ? const Center(child: Text("Eşleşen bileşen bulunamadı."))
                  : ListView.builder(
                      itemCount: results.length,
                      itemBuilder: (context, index) {
                        final Map<String, dynamic> item =
                            results[index] as Map<String, dynamic>;

                        final bool isRestricted =
                            (item['is_restricted'] == true);

                        final num scoreNum = (item['score'] as num? ?? 0);
                        final String scoreText =
                            scoreNum.toDouble().toStringAsFixed(0);

                        return Card(
                          color: isRestricted
                              ? Colors.orange.shade50
                              : Colors.green.shade50,
                          child: ListTile(
                            title: Text(item['name'] ?? 'Bilinmeyen'),
                            subtitle: Text(
                              "Tip: ${item['kb_type']} (Skor: $scoreText%)",
                            ),
                            trailing: isRestricted
                                ? Icon(Icons.warning_amber_rounded,
                                    color: Colors.orange.shade800)
                                : Icon(Icons.check_circle_outline_rounded,
                                    color: Colors.green.shade800),
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      );
    },
  );
}
}