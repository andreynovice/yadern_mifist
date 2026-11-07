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
    outputs_osnet\embeddings.json

`embeddings.json` содержит сохранённые OSNet-векторы подтверждённых наблюдений.
Каждая запись связана с `camera`, `local_id` и номером `frame`; поле
`embedding` содержит нормированный L2-вектор OSNet (обычно 512 чисел). Это
позволяет повторно считать похожесть людей и проверять работу Re-ID без
повторного извлечения признаков. При запуске с `--no-reid` OSNet отключён, и
файл будет создан с `model: null` и пустым `records`.

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

## Проекция людей в мировые координаты

Файл `project_people_world.py` предназначен для одной камеры. Он запускает YOLO,
сохраняет исходные bbox и верхнюю центральную точку bbox (приближение положения
головы), а затем переводит эту точку в метры относительно теннисного стола.
Одновременно создаются JSON и видео сверху на белом фоне. Красная точка на видео
показывает положение человека на полу; рядом выводится его номер в текущем кадре.

Запуск для `camera_4_lots.mp4` из корня проекта:

    .venv\Scripts\python project_people_world.py `
      --source videos_proms\camera_4_lots.mp4 `
      --output world_projection_camera4 `
      --model yolo11m.pt `
      --step 5 `
      --confidence 0.15 `
      --image-size 1280 `
      --person-height 1.78 `
      --focal-length 1800

После завершения ищите файлы:

    world_projection_camera4\world_tracks.json
    world_projection_camera4\world_projection.mp4

Параметры мировой проекции:

| Параметр | Назначение |
|---|---|
| `--source` | Путь к входному видео |
| `--output` | Папка для JSON и MP4 |
| `--model` | Файл весов YOLO |
| `--step` | Обрабатывать каждый N-й кадр; `1` точнее, но медленнее |
| `--max-frames` | Ограничение числа исходных кадров; `0` — весь файл |
| `--confidence` | Минимальная уверенность детектора |
| `--image-size` | Размер входа YOLO |
| `--person-height` | Принятый рост человека в метрах; по умолчанию `1.78` |
| `--table-width` | Длина стола по оси X, по умолчанию `2.74` м |
| `--table-depth` | Ширина стола по оси Y, по умолчанию `1.525` м |
| `--table-height` | Высота столешницы над полом, по умолчанию `0.76` м |
| `--focal-length` | Приближённое фокусное расстояние в пикселях |
| `--table-points` | 8 чисел: `x1 y1 x2 y2 x3 y3 x4 y4` углов стола на изображении |
| `--map-width-px`, `--map-height-px` | Размер белого видео в пикселях |
| `--map-padding` | Запас вокруг найденных мировых точек в метрах |

Порядок `--table-points` должен соответствовать углам в мировых координатах:

    (0, 0, table_height)
    (table_width, 0, table_height)
    (table_width, table_depth, table_height)
    (0, table_depth, table_height)

Встроенный порядок для текущей камеры: `689 618 1603 499 1888 833 751 1081`.
Если порядок перепутан, карта будет отражённой или искажённой.

`head_world` в JSON — координаты верхней точки bbox на плоскости высоты
`person_height`. `floor_world` — вертикальная проекция этой же точки на пол
(`z=0`), и именно её координаты используются для точки на белом видео.
В JSON также записаны `coordinate_system`, матрица камеры и все параметры запуска.

Фокусное расстояние по умолчанию (`1800`) является оценкой. Для метрически точного
результата нужны реальные `fx`, `fy`, `cx`, `cy` и коэффициенты дисторсии камеры.
Проверяйте результат так:

    .venv\Scripts\python -c "import json; d=json.load(open('world_projection_camera4\\world_tracks.json', encoding='utf-8')); print(d['coordinate_system']); print(sum(len(f['people']) for f in d['frames']))"

Если число людей равно нулю, уменьшите `--confidence`, поставьте `--step 1` и
запустите на участке видео, где люди точно видны. Для быстрого теста используйте
`--max-frames 100`. Видео можно открыть обычным VLC или проигрывателем Windows.

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

## Web API и Frontend

В репозитории есть локальный web-MVP: FastAPI отдаёт результаты анализа и
статический интерфейс из каталога web. Установите зависимости из
requirements.txt, затем запустите:

    .venv\\Scripts\\python -m uvicorn server:app --reload

Откройте http://127.0.0.1:8000. Кнопка «Подключить результат» регистрирует
папку с report.json, а «Новый анализ» запускает существующий pipeline в
фоновом потоке. Для видео можно указать каталог data; анализатор принимает
форматы mp4, mkv, avi, mov и webm.

API предоставляет GET /api/jobs/{id}/report, /tracks, /cameras,
/videos/{camera}, /embeddings, /world и экспорт файлов. Мировая проекция
остаётся отдельным single-camera экспериментом: она подключается из папки,
содержащей world_tracks.json и world_projection.mp4, и не связывается с
глобальными ID основного трекера.
