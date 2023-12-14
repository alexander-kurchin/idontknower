**НейроНезнайка** — это умный голосовой помощник, который отвечает в стиле книжного Незнайки. Это результат работы кросс-функциональной команды-победительницы в хакатоне издательского холдинга «Эксмо-АСТ».

Проект был представлен публике 2 декабря 2023 года на книжной выставке-ярмарке non/fictio№25.

# Какая задача стояла

В своём первом варианте техническое задание гласило:
> Реализовать приложение, в котором посетитель получает от ведущего микрофон и задаёт вопрос персонажу (Незнайке), который находится на экране, а персонаж отвечает в своем стиле (по сути, это кастомизированный голосовой помощник).

Простор для творчества и импровизаций был огромен.

# Что было реализовано

[Дизайнеры](https://www.behance.net/gallery/186713433/Voice-Assistant-APP-AI-UX-UI-neznajka) провели UX-исследование с респондентами. Создали концепцию и совместно с фронтендером воплотили её в жизнь.

[На фронтенде](https://github.com/Nick-Voskoboinikov/neznaika-for-eksmo) были реализованы голосовые команды, запускающие процесс записи вопроса (конец записи триггерился по секундному молчанию), выведения в режим паузы и закрытия приложения. Добавлены горячие клавиши для управления приложением, в том числе для удаления последней пары вопрос-ответ.

[Дата-саентисты](https://github.com/AsiyatShch/Neznaika) использовали открытые и некоммерческие, но в то же время достаточно серьёзные решения для получения наиболее точного распознавания, релевантных ответов и качественной озвучки. Была выбрана «более хулиганистая» модель генерации ответов, которую дообучили на историях из книг Николая Носова.

Мне как бэкендеру было необходимо соединить Дизайн и Искусственный Интеллект. Мой основной фреймворк Django, но для данного проекта я посчитал его избыточным. Хотелось что-то более легковесное, и выбор пал на FastAPI.

Из-за ресурсоёмкости ДС-модулей не было возможности запускать проект на личном компьютере. Для презентации MVP мы нашли изящное решение: запускали ДС прямо в Google Colab и прокидывали тоннели через [ngrok](https://ngrok.com/).

# Презентация НейроНезнайки в ТАСС

[ТАСС: 22 НОЯ, 15:55. Выставка иллюстраций к произведениям Носова пройдет на ярмарке Non/fiction](https://tass.ru/kultura/19353719/amp):
> [...] Она также отметила, что юные гости смогут задать вопросы виртуальному Незнайке. Как пояснила продюсер проекта по искусственному интеллекту школы IT-профессий Skillfactory Маргарита Хабарова, нейронезнайка - голосовой помощник. "Он умеет распознавать речь и отвечать на заданные вопросы в своей определенной стилистике, иногда шутя", - сказала она, отметив, что свой голос Незнайке, созданном при помощи технологии искусственного интеллекта, предоставила чтец Анна Сказко. [...]

### Youtube: Маргарита Хабарова о нас на пресс-конференции в ТАСС
[![Youtube: Маргарита о нас на пресс-конференции в ТАСС](http://img.youtube.com/vi/3qsIgYh-njA/0.jpg)](http://www.youtube.com/watch?v=3qsIgYh-njA "Маргарита о нас на пресс-конференции в ТАСС")

# Деплой

Работа на выставке накладывала определённые ограничения: на таких массовых мероприятиях, как правило, есть проблемы с интернетом. Соответственно необходимо было локальное решение. Издательство предоставило подходящее железо: Core i7-11700K, DDR4 32Gb, SSD 250+1000Gb, Quadro RTX A4000.

Мы провели консультацию с DevOps-специалистом и решили идти по пути контейнеризации нашего приложения. Упаковывали так, как оно и было разбито на этапе MVP:
1) Бэкенд, отдающий фронтенд и реализующий REST API (app)
2) Модуль распознавания речи и генерации ответа (sr_tg)
3) Модуль озвучки (tts)

Сутки перед выставкой — это мой *курс молодого бойца Docker*. Кроме того, из-за высоких требований ДС мы, по сути, впервые собирали наше приложение, которое мы улучшали до последнего дня, так что по ходу развёртывания вылезали и исправлялись непредвиденные ошибки, дорабатывалась обработка текстового ответа и т.д. и т.п.

### Youtube: Демонстрация работы НейроНезнайки
[![Youtube: Демонстрация работы НейроНезнайки](http://img.youtube.com/vi/QBbSnxUNZHA/0.jpg)](http://www.youtube.com/watch?v=QBbSnxUNZHA "Демонстрация работы НейроНезнайки")

###  Оплошность
Для работы голосовых команд всё-таки был необходим интернет: на фронтенде Незнайка слушал и распознавал через Google Cloud Speech-to-Text API. Это было внедрено на заре проекта, когда ещё не было чёткого понимания тенических требований. Спас ситуацию USB-модем.

# Успех на выставке

Детям очень понравилось общаться с Незнайкой: интерфейс взаимодействия был интуитивен, а ответы остроумные.

![children](https://github.com/alexander-kurchin/pics-for-md/blob/main/idontknower/children.jpg)

# Идеи развития

- Создание полноценного общедоступного веб-приложения
- Адаптивный дизайн под разные устройства
- Сбор статистики работы ДС-модулей для анализа и совершенствования моделей
- Более гибкая архитектура приложения
- Потоковое распознавание речи (в том числе голосовых команд)
- Поддержание диалога
- Дополнительные персонажи

# Итоги

Это был мой первый опыт работы в такой большой (10 человек) кросс-функциональной команде.

Главное, что я получил — это ***софтскиллы***.

Умение работать в условиях неопределенности, быть гибким и адаптивным, а также важное понимание, что:
- люди могут по-разному интерпретировать одни и те же слова;
- в команде необходима согласованность действий;
- просить о помощи не стыдно;
- ещё один пункт, спросите меня о нём на собеседовании.

![cert](https://github.com/alexander-kurchin/pics-for-md/blob/main/idontknower/cert.jpg)
