# People Tracking

Поддерживаются исходные видео в форматах MKV и MP4. Остальные файлы в папке входных данных игнорируются.

Основной запуск проекта выполняется через web-интерфейс. Конвейер track_project.py запускается сервером и создаёт unified-результат с глобальными треками, проекциями и готовыми видео с bbox.

## Запуск web-интерфейса

Запустите: .venv\Scripts\python -m uvicorn server:app --host 127.0.0.1 --port 8000

Откройте http://127.0.0.1:8000. В форме «Новый анализ» можно выбрать папку видео, модель, устройство, шаг, confidence, Re-ID и отрезок времени. Пустое поле конца обрабатывает видео до конца; max_frames ограничивает число кадров внутри выбранного отрезка.

## Запуск unified-конвейера вручную

Пример: python track_project.py --videos data --output outputs_unified --model yolo11m.pt --device cpu --step 20 --start-sec 60 --end-sec 90

Результат содержит tracks_projection.json, embeddings.json, tracked/*_tracked.mp4 и projections/*_projection.mp4. В web-интерфейс подключаются только каталоги с unified JSON и каталогом tracked; legacy report.json не поддерживается. Видео из tracked уже содержат bbox, поэтому интерфейс не рисует поверх них дополнительный overlay.

Для быстрого теста можно добавить --max-frames 120, --step 10 и --imgsz 640. Отдельный render.py остаётся вспомогательным инструментом для legacy-отчётов.
