// Flutter'ın temel Material (Google tasarımı) kütüphanesini içeri aktarıyoruz.
// iOS yapıyor olsak da, uygulamanın "iskeleti" için bu en güçlü kütüphanedir.
import 'package:flutter/material.dart';

// Oluşturacağımız ekranı buraya "import" ediyoruz.
// Henüz oluşturmadık, birazdan oluşturacağız. Hata vermesi normal.
import 'package:skin_app/screens/user_info_screen.dart';
// Not: 'kullanici_bilgi_projesi' sizin proje adınız olmalı, VS Code otomatik tamamlar.

// Her Flutter uygulamasının ana giriş noktası 'main' fonksiyonudur.
void main() {
  // runApp, Flutter'a "Benim ana widget'ım bu, uygulamayı bununla başlat" deme yöntemidir.
  runApp(const MyApp());
}

// MyApp, bizim uygulamamızın kök (root) widget'ıdır.
// Bu widget'ın kendisi değişmez, o yüzden 'StatelessWidget' (Durumsuz Widget) kullanırız.
class MyApp extends StatelessWidget {
  // const: Bu widget'ın değişmez olduğunu belirtir (performans optimizasyonu).
  const MyApp({super.key});

  // 'build' metodu, Flutter'a bu widget'ın nasıl görüneceğini anlattığımız yerdir.
  @override
  Widget build(BuildContext context) {
    // MaterialApp: Uygulamamızın temel iskeletini sağlar.
    // Tema, navigasyon (sayfa geçişleri) ve daha fazlasını yönetir.
    return MaterialApp(
      // title: Görev yöneticisinde vs. görünen uygulama adı (iOS'ta çok kullanılmaz).
      title: 'Kullanıcı Bilgi Uygulaması',
      
      // theme: Uygulamanın genel renk şeması.
      theme: ThemeData(
        // Ana renk paletimiz. iOS'ta bile bu temel renkleri kullanmak iyidir.
        primarySwatch: Colors.blue, 
        // Modern Material 3 tasarım dilini kullanalım
        useMaterial3: true,
      ),

      // Sağ üst köşedeki sinir bozucu "DEBUG" yazısını kaldıralım.
      debugShowCheckedModeBanner: false,
      
      // home: Uygulama açıldığında kullanıcıya gösterilecek İLK ekran.
      // UserInfoScreen'i buraya bağlıyoruz.
      home: const UserInfoScreen(),
    );
  }
}