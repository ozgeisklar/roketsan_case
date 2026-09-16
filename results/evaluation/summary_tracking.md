# Tracking ile Zamansal Iyilestirme

GT: 64 kare, 133 kutu (Stockflue_Flyaround, Surenen_Pass_Trail_Running)

| Model / asama              | mAP@50   | mAP@50-95 | AP_Small | AP_Med   | Opt. P   | Opt. R   | Opt. F1  |
|----------------------------|----------|----------|----------|----------|----------|----------|----------|
| yolo11x_1536 / raw         | 0.760    | 0.529     | 0.401    | 0.611    | 0.838    | 0.660    | 0.738    |
| yolo11x_1536 / nms         | 0.793    | 0.529     | 0.403    | 0.615    | 0.922    | 0.710    | 0.802    |
| yolo11x_1536 / botsort     | 0.806    | 0.539     | 0.431    | 0.614    | 0.918    | 0.750    | 0.826    |
|----------------------------|----------|----------|----------|----------|----------|----------|----------|
| Stockflue_Flya / raw       | 0.484    | 0.250     | 0.248    | 0.262    | 0.727    | 0.450    | 0.556    |
| Stockflue_Flya / nms       | 0.498    | 0.243     | 0.218    | 0.271    | 0.758    | 0.470    | 0.580    |
| Stockflue_Flya / botsort   | 0.515    | 0.249     | 0.220    | 0.276    | 0.714    | 0.470    | 0.567    |
| Surenen_Pass_T / raw       | 0.947    | 0.701     | 0.521    | 0.820    | 0.959    | 0.870    | 0.912    |
| Surenen_Pass_T / nms       | 0.970    | 0.696     | 0.521    | 0.815    | 0.973    | 0.900    | 0.935    |
| Surenen_Pass_T / botsort   | 0.991    | 0.711     | 0.566    | 0.813    | 0.963    | 0.970    | 0.966    |
|----------------------------|----------|----------|----------|----------|----------|----------|----------|
| yolo26x_1536 / raw         | 0.749    | 0.510     | 0.380    | 0.598    | 0.827    | 0.680    | 0.746    |
| yolo26x_1536 / nms         | 0.769    | 0.503     | 0.374    | 0.593    | 0.865    | 0.720    | 0.786    |
| yolo26x_1536 / botsort     | 0.775    | 0.506     | 0.381    | 0.594    | 0.922    | 0.710    | 0.802    |
|----------------------------|----------|----------|----------|----------|----------|----------|----------|
| Stockflue_Flya / raw       | 0.492    | 0.247     | 0.203    | 0.288    | 0.821    | 0.430    | 0.564    |
| Stockflue_Flya / nms       | 0.521    | 0.250     | 0.201    | 0.291    | 0.885    | 0.430    | 0.579    |
| Stockflue_Flya / botsort   | 0.513    | 0.242     | 0.200    | 0.290    | 0.821    | 0.430    | 0.564    |
| Surenen_Pass_T / raw       | 0.916    | 0.674     | 0.494    | 0.793    | 0.955    | 0.800    | 0.871    |
| Surenen_Pass_T / nms       | 0.929    | 0.661     | 0.482    | 0.793    | 0.921    | 0.870    | 0.895    |
| Surenen_Pass_T / botsort   | 0.939    | 0.665     | 0.502    | 0.793    | 0.947    | 0.900    | 0.923    |
|----------------------------|----------|----------|----------|----------|----------|----------|----------|
| dino_full / raw            | 0.906    | 0.650     | 0.579    | 0.698    | 0.974    | 0.840    | 0.902    |
| dino_full / nms            | 0.896    | 0.638     | 0.569    | 0.684    | 0.974    | 0.840    | 0.902    |
| dino_full / botsort        | 0.915    | 0.641     | 0.563    | 0.690    | 0.930    | 0.900    | 0.915    |
|----------------------------|----------|----------|----------|----------|----------|----------|----------|
| Stockflue_Flya / raw       | 0.774    | 0.352     | 0.395    | 0.330    | 0.946    | 0.660    | 0.778    |
| Stockflue_Flya / nms       | 0.750    | 0.340     | 0.389    | 0.314    | 0.946    | 0.660    | 0.778    |
| Stockflue_Flya / botsort   | 0.801    | 0.345     | 0.393    | 0.321    | 0.854    | 0.770    | 0.810    |
| Surenen_Pass_T / raw       | 0.986    | 0.826     | 0.701    | 0.898    | 0.987    | 0.960    | 0.973    |
| Surenen_Pass_T / nms       | 0.989    | 0.815     | 0.685    | 0.898    | 0.987    | 0.960    | 0.973    |
| Surenen_Pass_T / botsort   | 0.991    | 0.812     | 0.677    | 0.898    | 0.975    | 0.980    | 0.978    |
|----------------------------|----------|----------|----------|----------|----------|----------|----------|
| dino_tiled / raw           | 0.914    | 0.665     | 0.579    | 0.724    | 0.958    | 0.860    | 0.907    |
| dino_tiled / nms           | 0.914    | 0.665     | 0.579    | 0.724    | 0.958    | 0.860    | 0.907    |
| dino_tiled / botsort       | 0.930    | 0.670     | 0.576    | 0.733    | 0.944    | 0.880    | 0.911    |
|----------------------------|----------|----------|----------|----------|----------|----------|----------|
| Stockflue_Flya / raw       | 0.788    | 0.416     | 0.419    | 0.421    | 0.974    | 0.690    | 0.808    |
| Stockflue_Flya / nms       | 0.788    | 0.416     | 0.419    | 0.421    | 0.974    | 0.690    | 0.808    |
| Stockflue_Flya / botsort   | 0.828    | 0.424     | 0.422    | 0.440    | 0.889    | 0.750    | 0.814    |
| Surenen_Pass_T / raw       | 0.989    | 0.815     | 0.685    | 0.898    | 0.987    | 0.960    | 0.973    |
| Surenen_Pass_T / nms       | 0.989    | 0.815     | 0.685    | 0.898    | 0.987    | 0.960    | 0.973    |
| Surenen_Pass_T / botsort   | 0.991    | 0.812     | 0.677    | 0.898    | 0.975    | 0.980    | 0.978    |
|----------------------------|----------|----------|----------|----------|----------|----------|----------|

## Notlar

- Kapi kontrolu yolo11x_1536: raw mAP@50 0.760, beklenen 0.760 -> GECTI
- yolo11x_1536: conf>=0.25 calisma noktasinda dedektorun kacirdigi 38 GT kutusunun 2 tanesi interpolasyonla geldi
- Kapi kontrolu yolo26x_1536: raw mAP@50 0.749, beklenen 0.749 -> GECTI
- yolo26x_1536: conf>=0.25 calisma noktasinda dedektorun kacirdigi 43 GT kutusunun 1 tanesi interpolasyonla geldi
- Kapi kontrolu dino_full: raw mAP@50 0.906, beklenen 0.906 -> GECTI
- dino_full: conf>=0.25 calisma noktasinda dedektorun kacirdigi 15 GT kutusunun 2 tanesi interpolasyonla geldi
- Kapi kontrolu dino_tiled: raw mAP@50 0.914, beklenen 0.914 -> GECTI
- dino_tiled: conf>=0.25 calisma noktasinda dedektorun kacirdigi 12 GT kutusunun 1 tanesi interpolasyonla geldi
- Orneklem kucuk (133 kutu); birkac kutuluk degisim oranlarda buyuk gorunuyor.
- Interpolasyon cift yonlu, yani nedensel degil: gercek zamanli bir sistemde yalnizca track_buffer ile geriye donuk doldurma mumkun olurdu.
- Eslesen kutularda koordinat ham tespitten aliniyor, Kalman tahmininden degil. Kalman kutusu kullanildiginda mAP@50 ayni kaliyor ama DINO'nun mAP@50-95'i 0.641'den 0.628'e dusuyordu: sureklilik kazanci lokalizasyon kaybina deger degil. YOLO'da tersi gecerli (gurultulu kutulari duzeltiyor), yani bu takas dedektorun lokalizasyon kalitesine bagli.

