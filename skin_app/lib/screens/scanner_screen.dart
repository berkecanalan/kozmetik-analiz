// Gerekli kütüphaneleri import ediyoruz
import 'package:flutter/material.dart';
// Terminalden kurduğumuz tarayıcı paketini import ediyoruz
import 'package:mobile_scanner/mobile_scanner.dart';

class ScannerScreen extends StatefulWidget {
  const ScannerScreen({super.key});

  @override
  State<ScannerScreen> createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen> {
  // Tarayıcının kendisini (örn: flash açma, kamera değiştirme) yönetmek için
  // bir kontrolcü (controller) oluşturuyoruz.
  final MobileScannerController _scannerController = MobileScannerController(
    // (Opsiyonel) Hangi kameranın açılacağını belirler (arka kamera)
    facing: CameraFacing.back,
  );

  // Tarama işlemi bir kere başarılı olunca tekrar tekrar tetiklenmesin diye
  // bir "bayrak" (flag) değişkeni kullanıyoruz.
  bool _isScanCompleted = false;

  // ÇOK ÖNEMLİ: Ekran kapandığında (dispose olduğunda) controller'ı
  // hafızadan silmeliyiz. Bu, hafıza sızıntılarını (memory leaks) önler.
  @override
  void dispose() {
    _scannerController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Barkod/QR Tara'),
        // (Opsiyonel) Kullanıcının flash'ı açması veya kamerayı değiştirmesi için
        // AppBar'a ekstra butonlar ekleyebiliriz.
        actions: [
          IconButton(
            icon: const Icon(Icons.flash_on),
            onPressed: () => _scannerController.toggleTorch(), // Flash'ı aç/kapa
          ),
          IconButton(
            icon: const Icon(Icons.cameraswitch),
            onPressed: () => _scannerController.switchCamera(), // Kamerayı değiştir
          ),
        ],
      ),
      body: Stack(
        // 'Stack' widget'ı, widget'ları üst üste bindirmemizi sağlar.
        // 1. Katman (en altta): Kamera görüntüsü
        // 2. Katman (ortada): Tarama çerçevesi
        // 3. Katman (en üstte): Yardım metni
        children: [
          // 1. Katman: Kamera Görüntüsü
          MobileScanner(
            controller: _scannerController, // Hangi kamerayı yöneteceğini söylüyoruz
            
            // onDetect: BU, PAKETİN EN ÖNEMLİ KISMIDIR.
            // Bir kod algılandığında bu fonksiyon otomatik olarak tetiklenir.
            onDetect: (capture) {
              // Eğer daha önce bir tarama yapılmadıysa... (Bayrağı kontrol et)
              if (!_isScanCompleted) {
                // Bayrağı hemen 'true' yap ki bu kod bloğu tekrar çalışmasın.
                setState(() {
                  _isScanCompleted = true;
                });

                // 'capture' nesnesinin içinden taranan barkodları al.
                final List<Barcode> barcodes = capture.barcodes;
                
                if (barcodes.isNotEmpty) {
                  // Taranan ilk veriyi al (genelde tek veri olur)
                  // .rawValue, taranan verinin ham halidir (örn: "https://google.com")
                  final String scannedData = barcodes.first.rawValue ?? "Bilinmeyen Veri";
                  
                  print('--- TARANAN VERİ BULUNDU ---');
                  print(scannedData);
                  print('-----------------------------');

                  // Tarama başarılı. Şimdi bu veriyi bir önceki ekrana (Form Ekranı)
                  // "geri göndererek" bu ekranı kapat.
                  // 'Navigator.pop' bir ekrandan geri dönmemizi sağlar.
                  // İkinci parametre (scannedData), geri döndürdüğümüz veridir.
                  Navigator.pop(context, scannedData);
                }
              }
            },
          ),
          
          // 2. Katman: Tarama Alanı için görsel bir çerçeve (Opsiyonel ama güzel)
          // Kullanıcıya nereye odaklanacağını gösterir.
          Center(
            child: Container(
              width: 250, // Çerçevenin genişliği
              height: 250, // Çerçevenin yüksekliği
              decoration: BoxDecoration(
                border: Border.all(color: Colors.red.shade400, width: 4), // Kırmızı çerçeve
                borderRadius: BorderRadius.circular(12), // Köşeleri yuvarlat
              ),
            ),
          ),

          // 3. Katman: Kullanıcıya bilgi veren bir metin
          Align(
            alignment: Alignment.bottomCenter,
            child: Container(
              padding: const EdgeInsets.all(16.0),
              color: Colors.black.withAlpha(102), // Yarı saydam siyah arka plan
              child: const Text(
                'Lütfen QR kodu veya barkodu çerçevenin içine getirin.',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                ),
                textAlign: TextAlign.center,
              ),
            ),
          )
        ],
      ),
    );
  }
}