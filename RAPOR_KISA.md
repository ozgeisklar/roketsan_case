# Drone Videolarında İnsan Tespiti ve Zamansal İyileştirme

## 1. Amaç ve Problem Tanımı

Bu çalışmanın amacı, drone videolarında **insan** sınıfını tespit edip görüntü içerisindeki konumunu bounding box olarak belirleyen bir bilgisayarlı görü prototipi geliştirmektir. Veri setinde hazır ground-truth bulunmadığı ve tüm videoları elle etiketlemek gerçekçi olmadığı için problem yalnızca model seçimi olarak değil, aynı zamanda **veri seçimi ve etiketleme stratejisi** olarak ele alınmıştır.

Hedef sınıf olarak insan seçildi. Bunun nedeni drone videolarında insanların çoğu zaman küçük, uzak, hareketli ve arka planla karışabilir durumda görünmesidir. Bu özellikler nesne tespiti açısından anlamlı bir zorluk oluşturur.

## 2. Kullanılan Veri ve Video Seçimi

Çalışmada Kaggle Drone Videos veri seti kullanıldı. Önce arşivdeki videolar analiz edilerek çözünürlük, toplam kare sayısı, insan görülen kare oranı, kutu boyutu dağılımı ve küçük nesne oranı çıkarıldı. Bu analiz için `analyze_video_inventory.py`, sonuçlar için `results/video_inventory_summary.md` kullanıldı.

Test ve eğitim ayrımı video bazlı yapıldı. Test tarafında özellikle küçük nesne oranı yüksek ve takip açısından anlamlı iki video öne çıkarıldı:

- `Stockflue Flyaround.mp4`: 1280x720 çözünürlükte, insan kutularının yaklaşık %59.6'sı COCO small kategorisinde. Kamera hareketi ve küçük hedefler nedeniyle zor bir senaryo.
- `Surenen Pass Trail Running.mp4`: 406x720 çözünürlükte, koşan insan içeren dinamik bir takip videosu. İnsan kutularının yaklaşık %49.5'i small kategorisinde.

Test seti için videolardan her 30 karede bir frame alındı. Böylece 30 FPS videolarda yaklaşık saniyede bir örnekleme yapılmış oldu. Her frame'i etiketlemek yerine seyrek örnekleme tercih edildi; çünkü ardışık kareler büyük ölçüde benzer bilgi taşıyor ve tüm kareleri etiketlemek maliyeti ciddi şekilde artırıyor.

Filtered test setinde toplam **64 kare ve 133 insan kutusu** kullanıldı:

- `Stockflue Flyaround`: 24 kare, 53 kutu
- `Surenen Pass Trail Running`: 40 kare, 80 kutu

Eğitim tarafında kalan videolardan aktif öğrenme ile seçilen **93 kare ve 445 kutu** kullanıldı. Toplam elle etiketlenen veri miktarı, tüm video arşivine kıyasla oldukça sınırlı tutuldu.

## 3. Veri Hazırlama ve Etiketleme Stratejisi

Test verisi `prepare_test_set.py` ile hazırlandı. Script, seçilen videolardan belirli aralıklarla frame örnekledi ve bu frameler için Grounding DINO ile ön etiketler üretti. Ön etiketler CVAT formatına çevrildi, ardından CVAT üzerinde elle kontrol edilip düzeltildi.

Ön etiketleme aşamasında amaç yüksek precision değil, yüksek recall elde etmekti. Yani modelin fazla kutu üretmesi kabul edilebilir görüldü; çünkü yanlış kutuları silmek, sıfırdan kutu çizmekten daha hızlıdır. Bu yaklaşım, manuel etiketleme maliyetini düşürdü.

## 4. Başlangıç Yaklaşımı

Başlangıç yaklaşımı olarak COCO üzerinde önceden eğitilmiş YOLO ve Grounding DINO modelleri doğrudan test edildi. YOLO için `yolo11x` ve `yolo26x`, DINO için ise `IDEA-Research/grounding-dino-base` kullanıldı.

İlk deneylerde iki temel gözlem ortaya çıktı:

1. Girdi çözünürlüğü performansı ciddi şekilde etkiliyor. YOLO11x modelinde `imgsz=640` ile mAP@50 0.135 iken, `imgsz=1536` ile 0.477 seviyesine çıktı.
2. Küçük insanları tespit etmek en önemli hata kaynağıydı. Özellikle drone videolarında insan çok az piksel kapladığında modelin kaçırma oranı yükseldi.

Tam test değerlendirmesinde öne çıkan bazı baseline sonuçları:

| Model | mAP@50 | mAP@50-95 | AP_Small | FPS |
|---|---:|---:|---:|---:|
| YOLO11x 640 | 0.135 | 0.089 | 0.029 | 11.3 |
| YOLO11x 1536 | 0.477 | 0.266 | 0.207 | 2.5 |
| YOLO26x 1536 | 0.535 | 0.300 | 0.249 | 2.5 |
| DINO full | 0.418 | 0.247 | 0.201 | 0.6 |
| DINO tiled | 0.622 | 0.549 | 0.570 | 0.1 |

Bu sonuçlar küçük nesne problemi için yalnızca model değiştirmekten çok, çıkarım stratejisinin de önemli olduğunu gösterdi.

## 5. Alternatif Yaklaşım 1: Döşemeli Grounding DINO

Grounding DINO standart kullanımda görüntüyü modele vermeden önce yeniden boyutlandırıyor. Drone videolarında hedefler zaten küçük olduğu için bu yeniden boyutlandırma insanları daha da küçültüyor. Bu nedenle `tiled_dino.py` ile döşemeli çıkarım geliştirildi.

Bu yöntemde görüntü örtüşen parçalara ayrılıyor, her parça ayrı ayrı modele veriliyor ve çıkan kutular tekrar global koordinatlara taşınıyor. Daha sonra NMS ve kapsama bastırması ile tekrar kutular temizleniyor.

Bu yaklaşımın amacı, küçük insanları modele daha büyük ve görünür şekilde sunmaktı. Sonuç olarak DINO full ile mAP@50 0.418 iken, döşemeli DINO ile 0.622'ye çıktı. AP_Small değeri 0.201'den 0.570'e yükseldi. Bu, küçük nesne probleminde en büyük kazanımın döşemeli çıkarımdan geldiğini gösterdi.

Dezavantajı ise hesaplama maliyetidir. Döşemeli DINO daha doğru sonuç verirken FPS değeri belirgin şekilde düştü.

## 6. Alternatif Yaklaşım 2: Aktif Öğrenme ve Fine-tune

Tüm eğitim videolarını elle etiketlemek yerine aktif öğrenme kullanıldı. Amaç, model için en bilgilendirici kareleri seçip yalnızca bu kareleri elle etiketlemekti.

`kaggle_active_learning_selection.ipynb` içinde her aday kare için iki model çalıştırıldı:

- Grounding DINO tiled: öğretmen model
- YOLO: öğrenci model

Seçim skoru şu kriterlere dayandı:

- DINO'nun bulduğu ama YOLO'nun kaçırdığı kişiler
- Küçük kutular
- Düşük/orta güvenli belirsiz DINO tespitleri
- YOLO'nun tek başına ürettiği şüpheli kutular
- Kalabalık sahneler

Buna ek olarak aynı videodan birbirine çok yakın karelerin seçilmesini engelleyen bir çeşitlilik filtresi kullanıldı. Böylece ardışık ve neredeyse aynı frame'leri etiketlemek yerine farklı zamanlardan daha bilgilendirici örnekler seçildi.

Aktif öğrenme iki tur uygulandı. İlk turdan sonra fine-tune edilen model, ikinci turda yeniden öğrenci model olarak kullanıldı. Böylece ikinci tur seçimi güncel modelin hâlâ zorlandığı karelere göre yapıldı.

Fine-tune sonucunda filtered test setinde:

| Model | Zero-shot | Round 1 | Round 2 |
|---|---:|---:|---:|
| YOLO11x 1536 | 0.643 | 0.745 | 0.760 |
| YOLO26x 1536 | 0.737 | 0.758 | 0.749 |
| DINO full | 0.896 | 0.898 | 0.906 |
| DINO tiled | 0.875 | 0.914 | 0.914 |

YOLO11x tarafında belirgin iyileşme görüldü. DINO modelleri zaten güçlü başladığı için fine-tune kazancı daha sınırlı kaldı.

Eğitim sırasında önemli bir hata da tespit edildi: YOLO eğitiminde `single_cls=True` kullanıldığında pretrained sınıflandırma kafası bozulabiliyor ve model person sınıfını küçük veriyle yeniden öğrenmek zorunda kalıyor. Bu nedenle `single_cls=True` kaldırıldı ve COCO'nun 80 sınıf ismi `data.yaml` içine yazıldı. Ayrıca `optimizer=AdamW`, düşük öğrenme oranı ve `freeze=23` ile eğitim daha stabil hale getirildi.

## 7. Alternatif Yaklaşım 3: Zamansal Tracking

Dedektörler her frame'i bağımsız değerlendirir. Ancak videoda aynı insan genellikle ardışık karelerde görünmeye devam eder. Bu nedenle tracking ile zamansal bilgi kullanıldı.

Tracking için `tracking/detect_cache.py`, `tracking/offline_tracker.py`, `tracking/evaluate_tracking.py` ve `tracking/render_demo.py` dosyaları oluşturuldu.

Akış şu şekildedir:

1. Dedektör her video karesinde çalıştırılır ve ham kutular JSON olarak önbelleğe alınır.
2. BoT-SORT tracker, bu cache'lenmiş kutuları ve video framelerini sırayla alır.
3. Aynı kişiye ait kutular track ID ile bağlanır.
4. Kısa süreli detection boşlukları interpolasyon ile doldurulur.
5. Track'in güvenilirliğine göre skorlar yeniden düzenlenir.
6. Sonuçlar mevcut COCO ground-truth frameleri üzerinde değerlendirilir.

Detection ve tracking adımları ayrıldı çünkü detector çalıştırmak pahalı, tracking ise ucuzdur. Bu sayede aynı detector çıktısı üzerinde raw, NMS ve tracking sonuçları adil şekilde karşılaştırılabildi.

Filtered test setinde tracking sonuçları:

| Model / Aşama | mAP@50 | mAP@50-95 | AP_Small | Opt. R | Opt. F1 |
|---|---:|---:|---:|---:|---:|
| YOLO11x raw | 0.760 | 0.529 | 0.401 | 0.660 | 0.738 |
| YOLO11x NMS | 0.793 | 0.529 | 0.403 | 0.710 | 0.802 |
| YOLO11x BoT-SORT | 0.806 | 0.539 | 0.431 | 0.750 | 0.826 |
| YOLO26x raw | 0.749 | 0.510 | 0.380 | 0.680 | 0.746 |
| YOLO26x BoT-SORT | 0.775 | 0.506 | 0.381 | 0.710 | 0.802 |
| DINO full raw | 0.906 | 0.650 | 0.579 | 0.840 | 0.902 |
| DINO full BoT-SORT | 0.915 | 0.641 | 0.563 | 0.900 | 0.915 |
| DINO tiled raw | 0.914 | 0.665 | 0.579 | 0.860 | 0.907 |
| DINO tiled BoT-SORT | 0.930 | 0.670 | 0.576 | 0.880 | 0.911 |

Tracking her modelde mAP@50 değerini artırdı. En anlamlı etki YOLO tarafında görüldü; çünkü YOLO'nun kaçırdığı kare sayısı DINO'ya göre daha fazlaydı.

## 8. Nicel Karşılaştırma ve Genel Sonuç

Çalışmanın ana sonucu, performansın tek bir model değişikliğiyle değil, veri seçimi ve çıkarım stratejisinin birlikte tasarlanmasıyla arttığıdır.

| Aşama | YOLO11x mAP@50 | DINO tiled mAP@50 |
|---|---:|---:|
| Başlangıç | 0.643 | 0.875 |
| Aktif öğrenme + fine-tune | 0.760 | 0.914 |
| Tracking sonrası | 0.806 | 0.930 |

En yüksek skor DINO tiled + BoT-SORT ile elde edildi. Ancak hız açısından YOLO11x + BoT-SORT daha uygulanabilir bir seçenektir. Bu nedenle iki farklı kullanım senaryosu ortaya çıktı:

- En yüksek doğruluk isteniyorsa: DINO tiled + BoT-SORT
- Daha hızlı ve uygulanabilir sistem isteniyorsa: YOLO11x + BoT-SORT

## 9. Hata Analizi

Başarılı durumlar:

- Tek veya az sayıda insanın bulunduğu, hareketin daha düzenli olduğu sahnelerde hem DINO hem YOLO iyi çalıştı.
- Döşemeli DINO küçük insanları belirgin şekilde daha iyi yakaladı.
- Tracking, kısa süreli kaçırmaları ve düşük güvenli ama tutarlı tespitleri güçlendirdi.

Zayıf durumlar:

- `Stockflue Flyaround` videosunda küçük hedefler ve kamera hareketi nedeniyle YOLO hâlâ zorlandı.
- DINO tiled daha başarılı olsa da ciddi hesaplama maliyeti getirdi.
- Tracking, detector tamamen yanlış veya hiç tespit üretmiyorsa tek başına çözüm olamıyor; yalnızca önceki/sonraki karelerde görülen hedefleri taşıyabiliyor.

Belirsiz durumlar:

- Test seti 64 kare ve 133 kutu ile sınırlı olduğu için birkaç kutuluk fark metriklerde büyük görünebilir.
- Tracking metrikleri MOT metrikleriyle değil COCO metrikleriyle değerlendirildi; çünkü ground-truth içinde track ID bilgisi bulunmuyor.

## 10. Metrik Seçimi

Değerlendirmede COCO metrikleri kullanıldı:

- `mAP@50`: kutunun nesneyle yeterli örtüşmesini ölçen anlaşılır ana metrik
- `mAP@50-95`: lokalizasyon kalitesini daha sıkı değerlendiren metrik
- `AP_Small`: drone videolarında kritik olan küçük nesne performansı
- Precision, recall ve F1: pratik çalışma noktasındaki davranışı görmek için
- FPS / saniye başına kare: doğruluk ve hesaplama maliyeti dengesini göstermek için

MOTA, IDF1 gibi tracking metrikleri kullanılmadı; çünkü test etiketlerinde kişi kimliği/track ID bilgisi bulunmuyor.

## 11. Hesaplama Maliyeti

Filtered test videolarında yoğun çıkarım maliyeti yaklaşık olarak:

| Model | Süre / kare | Yaklaşık FPS |
|---|---:|---:|
| YOLO11x 1536 | 0.43 - 0.47 sn | 2.5 |
| YOLO26x 1536 | 0.44 - 0.49 sn | 2.4 |
| DINO full | 1.74 sn | 0.6 |
| DINO tiled | 1.65 - 8.57 sn | 0.1 - 0.3 |

Tracking aşaması detector'a göre çok daha ucuzdur. Bu nedenle detection çıktıları önce cache'lendi, ardından tracking bu cache üzerinde çalıştırıldı.

## 12. Güçlü ve Zayıf Yönler

Güçlü yönler:

- Tüm veri setini etiketlemek yerine aktif öğrenme ile sınırlı sayıda kare seçildi.
- Başlangıç modeli, döşemeli çıkarım, fine-tune ve tracking ayrı ayrı değerlendirildi.
- Küçük nesne problemi özel olarak analiz edildi.
- Sonuçlar hem nicel tablolarla hem demo videosuyla gösterilebilir hale getirildi.

Zayıf yönler:

- Test seti sınırlı boyutta.
- Fine-tune checkpoint seçimi ayrı bir validation seti yerine test sonuçlarına bakılarak yapıldı; bu, sonuçlarda iyimserlik oluşturabilir.
- DINO tiled yüksek doğruluk sağlasa da gerçek zamanlı kullanım için yavaş.
- Tracking çevrimdışı çalıştığında gelecek kare bilgisini kullanabildiği için gerçek zamanlı senaryoya birebir karşılık gelmez.

## 13. Çalıştırma Adımları

```bash
# Envanter analizi
python3 analyze_video_inventory.py

# Test kareleri ve DINO ön etiketleri
python3 prepare_test_set.py

# Baseline / fine-tune değerlendirme
python3 evaluate_on_gt.py --models yolo11x_1536,yolo26x_1536,dino_full,dino_tiled

# Tracking için yoğun detection cache
python3 tracking/detect_cache.py \
  --models yolo11x_1536,yolo26x_1536,dino_full,dino_tiled \
  --videos Stockflue_Flyaround,Surenen_Pass_Trail_Running

# Tracking çıktıları
python3 tracking/offline_tracker.py \
  --models yolo11x_1536,yolo26x_1536,dino_full,dino_tiled \
  --stages raw,nms,botsort

# Tracking değerlendirmesi
python3 tracking/evaluate_tracking.py \
  --models yolo11x_1536,yolo26x_1536,dino_full,dino_tiled \
  --stages raw,nms,botsort

# Demo videosu
python3 tracking/render_demo.py --slug Stockflue_Flyaround --model yolo11x_1536
```

## 14. Kaynaklar

- Kaggle Drone Videos veri seti
- Grounding DINO (`IDEA-Research/grounding-dino-base`)
- Ultralytics YOLO
- BoT-SORT
- CVAT
- COCO evaluation / `pycocotools`

