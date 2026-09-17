# Drone Videolarında İnsan Tespiti ve Zamansal İyileştirme Raporu

## 1. Amaç

Bu çalışmanın amacı, drone videolarında **insan** sınıfını tespit edip görüntü içerisindeki konumunu bounding box olarak belirleyen bir bilgisayarlı görü prototipi geliştirmektir. Veri setinde hazır ground-truth bulunmadığı ve tüm videoları elle etiketlemek gerçekçi olmadığı için problem yalnızca model seçimi olarak değil, aynı zamanda **veri hazırlama, aktif öğrenme ve zamansal iyileştirme** problemi olarak ele alındı.

Hedef sınıf olarak insan seçildi. Verilen datasette en çok tespit edilebilecek örnek sayısı insan class'ı için mevcut. Aynı zamanda drone videolarında insanlar çoğu zaman küçük, uzak, hareketli ve arka planla karışabilir durumda görünüyor. Bu da hem detection hem lokalizasyon açısından anlamlı bir zorluk oluşturuyor.

## 2. Veri Seçimi ve Etiketleme Stratejisi

Önce veri setindeki videolar `analyze_video_inventory.py` ile analiz edildi. Bu analizde çözünürlük, toplam kare sayısı, tespit yoğunluğu, kutu boyutu dağılımı ve COCO small/medium/large oranları çıkarıldı. Sonuçlar `results/video_inventory_summary.md` içinde tutuldu.

Test tarafında özellikle iki video öne çıkarıldı:

- `Stockflue Flyaround.mp4`: 1280x720 çözünürlükte, insan kutularının yaklaşık %59.6'sı COCO small kategorisinde. Küçük hedefler ve kamera hareketi nedeniyle zor bir sahne.
- `Surenen Pass Trail Running.mp4`: 406x720 çözünürlükte, koşan insan içeren dinamik bir takip sahnesi. İnsan kutularının yaklaşık %49.5'i COCO small kategorisinde.

Test için her 30 karede bir frame alındı. Bunun nedeni 30 FPS videolarda ardışık karelerin çok benzer bilgi taşımasıdır. Her frame'i etiketlemek test setini sayısal olarak büyütse de bilgi çeşitliliğini aynı oranda artırmaz ve manuel etiketleme maliyetini gereksiz yükseltir.

Filtered test seti:

| Video | Test frame | GT bbox |
|---|---:|---:|
| `Stockflue Flyaround.mp4` | 24 | 53 |
| `Surenen Pass Trail Running.mp4` | 40 | 80 |
| **Toplam** | **64** | **133** |

Test etiketleri `prepare_test_set.py` ile hazırlandı. Script önce seyrek frameleri kaydetti, ardından Grounding DINO ile ön etiket üretti. Bu ön etiketler CVAT'a aktarıldı ve elle düzeltilerek COCO formatında ground-truth elde edildi.

Ön etiketleme yüksek recall mantığıyla yapıldı. Yanlış pozitif kutuları silmek, sıfırdan kutu çizmekten daha hızlı olduğu için DINO eşiği kontrollü şekilde düşük tutuldu.

## 3. Başlangıç Yaklaşımı: Baseline Modeller

Başlangıçta COCO üzerinde eğitilmiş YOLO ve Grounding DINO modelleri fine-tune edilmeden test edildi. YOLO tarafında `yolo11x` ve `yolo26x`, DINO tarafında `IDEA-Research/grounding-dino-base` kullanıldı.

İlk baseline değerlendirmesi:

| Model           | mAP@50   | mAP@50-95 | AP_Small | AP_Med   | Opt. P   | Opt. R   | Opt. F1  | Sure (s) | FPS   |
|-----------------|----------|-----------|----------|----------|----------|----------|----------|----------|-------|
| yolo11x_640     | 0.548    | 0.381     | 0.129    | 0.547    | 0.849    | 0.460    | 0.597    | 0.095    | 10.5  |
| yolo11x_1536    | 0.643    | 0.455     | 0.281    | 0.576    | 0.802    | 0.600    | 0.686    | 0.392    | 2.6   |
| yolo26x_640     | 0.695    | 0.477     | 0.271    | 0.590    | 0.909    | 0.600    | 0.723    | 0.080    | 12.6  |
| yolo26x_1536    | 0.737    | 0.503     | 0.354    | 0.590    | 0.854    | 0.660    | 0.745    | 0.406    | 2.5   |
| dino_full       | 0.896    | 0.695     | 0.634    | 0.727    | 0.982    | 0.810    | 0.888    | 1.534    | 0.7   |
| dino_tiled      | 0.875    | 0.724     | 0.637    | 0.769    | 0.940    | 0.810    | 0.870    | 3.845    | 0.3   |


Bu tabloda en önemli gözlem, küçük nesne performansının girdi çözünürlüğüne çok duyarlı olmasıdır. YOLO11x modelinde 640 çözünürlükten 1536 çözünürlüğe çıkınca mAP@50 0.548'den 0.643'e yükseldi. Benzer şekilde YOLO26x için 0.695'ten 0.737'ye çıktı.

Bu yüzden sonraki deneylerde YOLO'nun 640 varyantları ayrıca fine-tune edilmedi. Çünkü 1536 varyantları baseline aşamasında hem YOLO11x hem YOLO26x için açık şekilde daha iyi performans verdi. Eğitim bütçesi daha anlamlı olan 1536 çözünürlükteki modellere ayrıldı.

## 4. Filtered Test Seti ile Baseline Tekrarı

Filtered test seti üzerinde baseline modeller tekrar değerlendirildi. Bu tablo, fine-tune ve tracking aşamalarının karşılaştırıldığı ana referans olarak kullanıldı.

| Model | mAP@50 | mAP@50-95 | AP_Small | AP_Med | Opt. P | Opt. R | Opt. F1 | Süre (s) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo11x_1536_filtered | 0.643 | 0.455 | 0.281 | 0.576 | 0.802 | 0.600 | 0.686 | 0.409 | 2.4 |
| yolo26x_1536_filtered | 0.737 | 0.503 | 0.354 | 0.590 | 0.854 | 0.660 | 0.745 | 0.413 | 2.4 |
| dino_full_filtered | 0.896 | 0.695 | 0.634 | 0.727 | 0.982 | 0.810 | 0.888 | 1.543 | 0.6 |
| dino_tiled_filtered | 0.875 | 0.724 | 0.637 | 0.769 | 0.940 | 0.810 | 0.870 | 3.879 | 0.3 |

Filtered test setinde DINO modellerinin güçlü başladığı görülüyor. YOLO tarafında ise özellikle `yolo11x_1536` hâlâ iyileştirmeye açık durumdaydı.

## 5. Alternatif Yaklaşım 1: Döşemeli Grounding DINO

Grounding DINO standart kullanımda görüntüyü modele vermeden önce yeniden boyutlandırıyor. Drone videolarında insanlar zaten küçük olduğu için bu yeniden boyutlandırma uzak hedefleri daha da küçültüyor. Bu nedenle `tiled_dino.py` içinde döşemeli çıkarım geliştirildi.

Bu yöntemde görüntü örtüşen parçalara ayrılıyor, her parça ayrı ayrı DINO'ya veriliyor, çıkan kutular global koordinata taşınıyor ve NMS/kapsama bastırması ile birleştiriliyor.

## 6. Alternatif Yaklaşım 2: Aktif Öğrenme ve Fine-tune

Tüm eğitim videolarını elle etiketlemek yerine aktif öğrenme uygulandı. Amaç, en bilgilendirici frameleri seçip yalnızca bu frameleri elle etiketlemekti.

`kaggle_active_learning_selection.ipynb` içinde her aday frame için iki model çalıştırıldı:

- Grounding DINO tiled: öğretmen model
- YOLO: öğrenci model

Seçim skoru şu fikirlere dayandı:

- DINO'nun bulduğu ama YOLO'nun kaçırdığı insanlar
- Küçük kutular
- Düşük/orta güvenli belirsiz DINO tespitleri
- YOLO'nun tek başına ürettiği şüpheli kutular
- Kalabalık sahneler

Bu skora ek olarak çeşitlilik filtresi kullanıldı. Aynı videodan birbirine çok yakın frameler seçilmedi; böylece birbirinin neredeyse kopyası olan karelere etiketleme bütçesi harcanmadı.

Aktif öğrenme iki tur halinde uygulandı:

| Tur | Toplam frame | Açıklama |
|---|---:|---|
| Round 1 | 93 | İlk aktif öğrenme seçimi; DINO tiled ve YOLO uyuşmazlığına göre seçilip elle düzeltildi. |
| Round 2 | 40 | Round 1 sonrası fine-tune edilen model yeniden öğrenci olarak kullanıldı; önceki turdaki kareler havuzdan çıkarıldı. |
| **Toplam** | **133** | Nihai aktif öğrenme eğitim seti. |

Nihai aktif öğrenme veri setinde toplam **133 frame ve 639 bbox** bulunuyor. Eğitim notebook'larında split video prefix'e göre yapıldı:

| Split | Frame | Bbox |
|---|---:|---:|
| Train | 108 | 580 |
| Validation | 24 | 59 |
| **Toplam** | **133** | **639** |

Not: Yerel COCO annotation dosyasında round bilgisi ayrı bir alan olarak tutulmadığı için Round 2'de eklenen 43 frame'in train/validation kırılımı doğrudan dosyadan ayrıştırılamıyor. Bu nedenle raporda Round 2 toplam ekleme ve nihai train/validation dağılımı birlikte verildi.

YOLO fine-tune sırasında iki önemli eğitim problemi çözüldü:

- `single_cls=True` kaldırıldı. Bu ayar Ultralytics içinde `nc=1` zorladığı için pretrained sınıflandırma kafasının transferini bozabiliyordu.
- `data.yaml` içine COCO'nun 80 sınıf ismi yazıldı. Böylece pretrained person bilgisi korunmuş oldu.
- `optimizer=AdamW`, düşük öğrenme oranı, `freeze=23`, `warmup_epochs=5` gibi ayarlarla küçük veri üzerinde eğitim daha stabil hale getirildi.

YOLO eğitim ayarları: `epochs=50`, `imgsz=1536`, `batch=4`, `optimizer=AdamW`, `lr0=0.0001`, `freeze=23`, `save_period=5`.

DINO fine-tune ayarları: `epochs=50`, `LR=1e-5`, `GRAD_ACCUM=4`, `TILE=True`.

## 7. Fine-tune Sonuçları: Round 1

Round 1 sonrası filtered test seti sonuçları:

| Model | mAP@50 | mAP@50-95 | AP_Small | AP_Med | Opt. P | Opt. R | Opt. F1 | Süre (s) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo11x_1536_filtered | 0.745 | 0.515 | 0.406 | 0.593 | 0.754 | 0.690 | 0.721 | 0.405 | 2.5 |
| yolo26x_1536_filtered | 0.758 | 0.510 | 0.377 | 0.597 | 0.852 | 0.690 | 0.762 | 0.410 | 2.4 |
| dino_full_filtered | 0.898 | 0.641 | 0.580 | 0.686 | 0.982 | 0.840 | 0.906 | 1.559 | 0.6 |
| dino_tiled_filtered | 0.914 | 0.648 | 0.577 | 0.700 | 0.983 | 0.850 | 0.912 | 4.029 | 0.2 |

Round 1 özellikle YOLO tarafında belirgin iyileşme sağladı. `yolo11x_1536_filtered` mAP@50 değeri 0.643'ten 0.745'e çıktı.

## 8. Fine-tune Sonuçları: Round 2

Round 2 sonrası filtered test seti sonuçları:

| Model | mAP@50 | mAP@50-95 | AP_Small | AP_Med | Opt. P | Opt. R | Opt. F1 | Süre (s) | FPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| yolo11x_1536_filtered | 0.760 | 0.529 | 0.401 | 0.611 | 0.838 | 0.660 | 0.738 | 0.401 | 2.5 |
| yolo26x_1536_filtered | 0.749 | 0.510 | 0.380 | 0.599 | 0.827 | 0.680 | 0.746 | 0.408 | 2.4 |
| dino_full_filtered | 0.906 | 0.650 | 0.579 | 0.698 | 0.974 | 0.840 | 0.902 | 1.551 | 0.6 |
| dino_tiled_filtered | 0.914 | 0.664 | 0.579 | 0.724 | 0.958 | 0.860 | 0.907 | 3.847 | 0.3 |

Round 2'de `yolo11x` ve `dino_full` tarafında küçük bir artış görüldü. `yolo26x` çok küçük bir düşüş gösterdi; filtered test seti 133 kutu ile sınırlı olduğu için bu fark birkaç kutuluk değişimden etkilenebilir.

## 9. Alternatif Yaklaşım 3: Zamansal Tracking

Dedektörler her frame'i bağımsız değerlendirir. Ancak videoda aynı insan genellikle ardışık karelerde görünmeye devam eder. Bu nedenle son aşamada BoT-SORT tabanlı zamansal tracking kullanıldı.

BoT-SORT'un seçilme nedeni, klasik IoU/Kalman tabanlı takip yöntemlerine ek olarak **global motion compensation (GMC)** desteği sunmasıdır. Drone videolarında yalnızca hedef insan hareket etmez; kamera da sürekli döner, yaklaşır, uzaklaşır veya yana kayar. Bu durumda ardışık iki frame arasında bütün sahne hareket ettiği için sabit kamera varsayımına dayalı takipçiler aynı kişiyi yanlış konuma taşınmış gibi görebilir. BoT-SORT içindeki `gmc_method: sparseOptFlow` ayarı, arka plandaki genel kamera hareketini optik akışla tahmin ederek kutu eşleştirmesinden önce bu hareketi telafi eder. Bu nedenle drone videosu gibi ego-hareketin yüksek olduğu bir problemde BoT-SORT, yalnızca IoU eşleştirmesi yapan daha basit takipçilere göre daha uygun görüldü.

Bu çalışmada BoT-SORT yeni bir detector olarak kullanılmadı. Detector çıktıları önce cache'lendi, ardından BoT-SORT bu kutuları ardışık frame'lerde aynı kişiye ait track'lere bağlamak için kullanıldı. Yani BoT-SORT'un görevi insanı ilk kez bulmak değil, zaten üretilmiş detection kutularının zamansal sürekliliğini kullanarak daha kararlı bir çıktı üretmektir.

Tracking pipeline:

1. `tracking/detect_cache.py`: Her frame için detector çıktısını JSON olarak cache'ler.
2. `tracking/offline_tracker.py`: Cache'lenmiş bbox'ları BoT-SORT'a verir, track ID üretir, kısa boşlukları interpolasyonla doldurur ve skorları track güvenine göre günceller.
3. `tracking/evaluate_tracking.py`: Tracking çıktısını GT framelerine geri eşler ve COCO metriklerini hesaplar.
4. `tracking/render_demo.py`: Ham detection ve tracking sonucunu yan yana demo videosu olarak üretir.

Detection ve tracking bilinçli olarak ayrıldı. Detector çalıştırmak pahalı, tracking ise çok daha ucuzdur. Ayrıca raw, NMS ve tracking sonuçlarının aynı detector cache'inden başlaması deney tasarımını daha adil hale getirir.

Raporda tracking için yalnızca final BoT-SORT sonucu gösterildi:

| Model | mAP@50 | mAP@50-95 | AP_Small | AP_Med | Opt. P | Opt. R | Opt. F1 | GT yakalama notu |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| yolo11x_1536 + BoT-SORT | 0.806 | 0.539 | 0.431 | 0.614 | 0.918 | 0.750 | 0.826 | conf>=0.25 noktasında 38 kaçan GT'nin 2'si interpolasyonla geldi |
| yolo26x_1536 + BoT-SORT | 0.775 | 0.506 | 0.381 | 0.594 | 0.922 | 0.710 | 0.802 | conf>=0.25 noktasında 43 kaçan GT'nin 1'i interpolasyonla geldi |
| dino_full + BoT-SORT | 0.915 | 0.641 | 0.563 | 0.690 | 0.930 | 0.900 | 0.915 | conf>=0.25 noktasında 15 kaçan GT'nin 2'si interpolasyonla geldi |
| dino_tiled + BoT-SORT | 0.930 | 0.670 | 0.576 | 0.733 | 0.944 | 0.880 | 0.911 | conf>=0.25 noktasında 12 kaçan GT'nin 1'i interpolasyonla geldi |

Bu bilgi şu şekilde de okunabilir:

| Model | Detector'ın yakaladığı GT | Interpolasyonla eklenen GT | Tracking sonrası yakalanan GT |
|---|---:|---:|---:|
| yolo11x_1536 | 95 / 133 | +2 | 97 / 133 |
| yolo26x_1536 | 90 / 133 | +1 | 91 / 133 |
| dino_full | 118 / 133 | +2 | 120 / 133 |
| dino_tiled | 121 / 133 | +1 | 122 / 133 |

Burada önemli nokta şudur: Tracking kazancı yalnızca interpolasyonla gelen kutulardan ibaret değildir. NMS, kısa track'lerin skorunu düşürme ve güvenilir track'lerin skorunu yükseltme de mAP@50 artışına katkı verdi.

## 10. Genel mAP@50 Karşılaştırması

Ana aşamaların mAP@50 karşılaştırması:

| Aşama | YOLO11x | YOLO26x | DINO full | DINO tiled |
|---|---:|---:|---:|---:|
| Filtered baseline | 0.643 | 0.737 | 0.896 | 0.875 |
| Fine-tune Round 1 | 0.745 | 0.758 | 0.898 | 0.914 |
| Fine-tune Round 2 | 0.760 | 0.749 | 0.906 | 0.914 |
| BoT-SORT sonrası | 0.806 | 0.775 | 0.915 | 0.930 |

En yüksek sonuç `dino_tiled + BoT-SORT` ile elde edildi. Ancak hız açısından `yolo11x_1536 + BoT-SORT` daha uygulanabilir bir prototip alternatifi olarak değerlendirilebilir.

## 11. Hata Analizi

Başarılı durumlar:

- Döşemeli DINO küçük insanları belirgin şekilde daha iyi yakaladı.
- Aktif öğrenme özellikle YOLO11x tarafında anlamlı performans artışı sağladı.
- Tracking, detector'ın tek karelik kaçırmalarını ve skor sıralamasındaki zayıflıkları iyileştirdi.
- `Surenen Pass Trail Running.mp4` gibi daha düzenli hareket içeren sahnelerde tracking ve DINO oldukça başarılı oldu.

Zayıf durumlar:

- `Stockflue Flyaround.mp4` hâlâ zorlayıcıdır. Küçük hedefler, kamera hareketi ve hedeflerin sahne içinde az piksel kaplaması YOLO tarafında performansı sınırladı.
- DINO tiled en başarılı yöntemlerden biri olsa da hesaplama maliyeti yüksektir.
- Tracking, detector hiç aday kutu üretmediğinde tek başına yeni nesne bulamaz; yalnızca zaman içinde görülen hedefleri daha tutarlı hale getirir.

Belirsiz durumlar:

- Filtered test seti 64 frame / 133 bbox ile sınırlıdır. Bu nedenle birkaç kutuluk fark metriklerde görünür değişim oluşturabilir.
- GT'de track ID olmadığı için MOTA/IDF1 gibi tracking metrikleri kullanılmadı. Tracking katkısı COCO detection metrikleri ve interpolasyonla geri kazanılan GT sayısı üzerinden ölçüldü.

## 12. Metrik Seçimi

COCO metrikleri tercih edildi:

- `mAP@50`: genel detection başarısını yorumlamak için ana metrik.
- `mAP@50-95`: daha sıkı lokalizasyon kalitesini ölçer.
- `AP_Small`: drone videolarındaki küçük insan problemi için kritik metrik.
- `AP_Medium`: orta boy hedeflerdeki davranışı ayırmak için.
- `Opt. Precision`, `Opt. Recall`, `Opt. F1`: pratik çalışma noktasındaki dengeyi göstermek için.
- Süre ve FPS: doğruluk-maliyet takasını göstermek için.

## 13. Hesaplama Maliyeti

Filtered test videoları üzerinde yoğun çıkarım süreleri yaklaşık olarak:

| Model | Süre / frame | Yaklaşık FPS |
|---|---:|---:|
| YOLO11x 1536 | 0.43 - 0.47 sn | 2.5 |
| YOLO26x 1536 | 0.44 - 0.49 sn | 2.4 |
| DINO full | 1.74 sn | 0.6 |
| DINO tiled | 1.65 - 8.57 sn | 0.1 - 0.3 |

Bu nedenle tracking aşamasında önce detection çıktıları cache'lendi. Daha sonra BoT-SORT bu cache üzerinde çalıştırıldı. Böylece pahalı detector adımı tekrar tekrar çalıştırılmadan tracking ve değerlendirme yapılabildi.

## 14. Güçlü ve Zayıf Yönler

Güçlü yönler:

- Tüm veri seti yerine sınırlı sayıda frame etiketlendi.
- Video seçimi ve train/test ayrımı veri analizine dayandırıldı.
- Baseline, tiled inference, active learning, fine-tune ve tracking ayrı ayrı ölçüldü.
- Her tespit için skor üretildi ve COCO metrikleriyle değerlendirildi.
- Demo videosu ile nitel sonuç gösterilebilir hale getirildi.

Zayıf yönler:

- Test seti sınırlı büyüklükte.
- DINO tiled yüksek doğruluk verse de gerçek zamanlı kullanım için yavaş.
- Offline tracking gelecek frame bilgisinden faydalanabildiği için gerçek zamanlı sistemle birebir aynı değildir.


## 15. Kullanılan Kaynaklar

- Kaggle Drone Videos veri seti
- Grounding DINO (`IDEA-Research/grounding-dino-base`)
- Ultralytics YOLO
- BoT-SORT
- CVAT
- COCO evaluation / `pycocotools`

