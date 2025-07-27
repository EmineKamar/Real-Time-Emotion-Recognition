# Real-Time-Emotion-Recognition
# Gemma Real-Time Emotion Recognition

Gerçek zamanlı ses tabanlı duygu tanıma sistemi.

## Özellikler

- Whisper tabanlı ses embedding çıkarımı
- Transformer tabanlı duygu sınıflandırma
- WebRTC VAD ile gerçek zamanlı konuşma tespiti
- Transition matrix ve Markov modelleri ile duygu tahminini geliştirme
- Offline ve online mod desteği

## Dosya Yapısı

- `data/` - Ham ve ön işlenmiş veri
- `models/` - Eğitilmiş modeller
- `src/` - Kaynak kodlar
- `train_transformer.py` - Transformer modeli eğitimi
- `real_time_emotion.py` - Canlı duygu tahmini uygulaması
- `vad_utils.py` - VAD yardımcı sınıfı
- `transition_matrix.py` - Geçiş matrisi hesaplamaları
- `whisper_utils.py` - Whisper embedding çıkarma yardımcıları
- `run_app` - Proje başlatma scripti

## Kurulum

Python 3.8+ ve gerekli kütüphaneler:

```bash
pip install -r requirements.txt
