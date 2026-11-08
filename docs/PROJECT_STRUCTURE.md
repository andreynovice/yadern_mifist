# Структура проекта

## Дерево шаблона hacathon_template-main

~~~text
hacathon_template-main/
├── .gitignore
├── README.md                         # пустой файл
├── artifacts/
│   ├── demo/README.md                # список материалов демонстрации
│   └── test-results/
│       ├── README.md                 # заготовка тестовых артефактов
│       └── results.csv               # демонстрационная строка-заглушка
├── config/example.env                # SAMPLE_RATE, SYMBOL_DURATION_MS, OUTPUT_DIRECTORY
├── docs/
│   ├── ARCHITECTURE.md               # пустые разделы архитектуры
│   ├── LIMITATIONS.md                # пустые разделы ограничений
│   ├── PROTOCOL.md                   # заготовка акустического протокола
│   └── TEST_RESULTS.md               # пустая заготовка тестов
└── src/README.md                     # placeholder платформенных реализаций
~~~

В шаблоне нет исходного кода в src, заполненного README, демонстрационных файлов, протокола передачи, тестов или реальных результатов измерений. results.csv содержит строку DELETE THIS EXAMPLE ROW и не является результатом тестирования.

## Фактическое приложение в корне репозитория

~~~text
.
├── server.py                         # FastAPI API и запуск заданий
├── track_project.py                  # основной unified-конвейер
├── analyze.py                        # ранний standalone-конвейер
├── app.py                            # старый Tkinter-просмотрщик
├── render.py                         # legacy-рендеринг bbox
├── detect_render.py                  # отдельный детектор и рендеринг
├── project_people_world.py            # проекция видео на мировую плоскость
├── camera_graph.json                 # граф переходов камер
├── camera_calibrations.json          # калибровка камер
├── office_map.json                   # карта помещения
├── yolo11m.pt                        # веса YOLO
├── requirements.txt                  # зависимости
├── data/                             # исходные видео, сейчас MKV
├── runs/                             # результаты запусков
├── web/                              # HTML, CSS и JavaScript интерфейса
└── hacathon_template-main/           # внешний шаблон документации
~~~

## Результат запуска

~~~text
runs/<N>/
├── tracks_projection.json             # unified-отчёт с глобальными треками
├── embeddings.json                    # признаки Re-ID
├── tracked/*_tracked.mp4              # видео с bbox
└── projections/*_projection.mp4      # видео проекции
~~~

Новые задания получают последовательные номера Эксперимент N и каталог runs/N.

## Поток данных

~~~text
data/*.mkv или data/*.mp4
        │
        ▼
server.py -- POST /api/jobs --> track_project.py
                                      │
                                      ├── YOLO и Deep SORT
                                      ├── OSNet или HSV признаки
                                      ├── объединение локальных сегментов
                                      ├── граф камер и калибровка
                                      └── H.264-видео с bbox
                                      │
                                      ▼
                              unified JSON и видео в runs/N
~~~
