# Трекинг людей по видео

Проект находит людей YOLO, строит локальные треки Deep SORT с OSNet Re-ID и сохраняет report.json, tracks.csv и видео с рамками.

## Установка

PowerShell:

    cd C:\Users\danii\projects\test
    py -3.12 -m venv .venv
    .venv\Scripts\python -m pip install --upgrade pip
    .venv\Scripts\python -m pip install -r requirements.txt

При первом запуске OSNet скачает свои веса. CPU поддерживается. Для GPU нужен CUDA PyTorch и параметр --device cuda.

## Основной запуск

Из корня проекта:

    .venv\Scripts\python analyze.py --videos videos --output outputs_osnet --model yolo11m.pt --step 5 --conf 0.15 --imgsz 1280 --gap 8 --device cpu

Результат:

    outputs_osnet\report.json
    outputs_osnet\tracks.csv

Для videos_proms:

    .venv\Scripts\python analyze.py --videos videos_proms --output outputs_proms --model yolo11m.pt --step 5 --conf 0.15 --imgsz 1280 --gap 8 --device cpu

В этой папке находится camera_4_lots.mp4. Если нужна только она, создайте папку с одним файлом:

    New-Item -ItemType Directory -Force input_camera4
    Copy-Item videos_proms\camera_4_lots.mp4 input_camera4\
    .venv\Scripts\python analyze.py --videos input_camera4 --output outputs_camera4 --model yolo11m.pt --step 5 --device cpu

## Быстрый тест

    .venv\Scripts\python analyze.py --videos videos_proms --output outputs_smoke --model yolo11m.pt --step 10 --max-frames 120 --imgsz 640 --device cpu

В report.json проверьте reid=osnet_x1_0, reid_device=cpu или cuda и правильное cameras[].frames.

## Отрисовка

После анализа используйте тот же каталог videos и соответствующий report:

    .venv\Scripts\python render.py --videos videos --report outputs_osnet\report.json --output rendered_osnet --hold 0

Для camera_4:

    .venv\Scripts\python render.py --videos input_camera4 --report outputs_camera4\report.json --output rendered_camera4 --hold 0

Результат — MP4 в указанной папке. hold=0 рисует bbox только на кадрах, сохранённых в отчёте. hold=5 удерживает bbox ещё пять кадров и влияет только на видео, не на жизнь трека.

## Параметры analyze.py

| Параметр | Смысл |
|---|---|
| --videos | Папка входных MP4 |
| --output | Папка report.json и tracks.csv |
| --model | Путь к YOLO-весам, обычно yolo11m.pt |
| --step | Каждый N-й кадр: 1 точнее, 5 быстрее |
| --conf | Минимальная уверенность YOLO; меньше даёт больше ложных детекций |
| --imgsz | Размер входа YOLO; больше точнее для мелких людей, но медленнее |
| --gap | Максимальный временной разрыв для межкамерного объединения, секунды |
| --max-frames | Лимит кадров на файл; 0 означает весь файл |
| --device | cpu или cuda |
| --no-reid | Отключает OSNet для baseline-сравнения |

## Параметры render.py

| Параметр | Смысл |
|---|---|
| --videos | Та же папка, что была указана при анализе |
| --report | Путь к report.json |
| --output | Папка итоговых MP4 |
| --hold | Сколько следующих кадров удерживать bbox; по умолчанию 0 |

## Что делает код

YOLO оставляет класс person. OSNet строит embedding внешности. Deep SORT с калмановским прогнозом и сопоставлением детекций создаёт локальный ID внутри камеры. После минимум трёх подтверждённых обновлений сегмент сохраняется. Сегменты разных камер объединяются по времени и похожести OSNet.

Три обновления означают три обработанных кадра. При step=5 это примерно 0.6 секунды при 25 FPS. Поэтому короткие появления могут быть удалены. Для короткой сцены используйте step=1 или step=2.

## Проверка

    $env:PYTHONDONTWRITEBYTECODE="1"
    .venv\Scripts\python -m py_compile analyze.py render.py detect_render.py app.py

Проверьте видео на сценах с пересечением людей, пропусками детектора, похожей одеждой и переходом между камерами. Сначала смотрите tracks.csv, затем rendered_*. Число total_tracks нельзя считать точным без ручной разметки.

## Ограничения и следующие задачи

- локальный трекер Deep SORT; Hybrid-SORT пока не подключён;
- нет карты разрешённых переходов между камерами;
- OSNet не дообучен на этих камерах;
- app.py пока экспериментальный;
- разные FPS камер нужно учитывать при синхронном просмотре;
- полный CPU-запуск может быть долгим.
