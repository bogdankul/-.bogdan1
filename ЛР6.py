АЛГОРИТМЫ СОРТИРОВКИ

СОРТИРОВКА ОБМЕНОМ
def bubble_sort(arr):
    
    n = len(arr)  # Получаем длину входящего массива
    
    # Цикл продолжается до тех пор, пока список не будет полностью отсортирован
    for i in range(n):
        # Переменная для отслеживания наличия перемещений на текущей итерации
        swapped = False
        
        # Обходим список от начала до конца (за исключением уже проверенных элементов)
        for j in range(0, n-i-1):
            
            # Если текущий элемент больше последующего, меняем их местами
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True  # Фиксируем факт свершившегося обмена
                
        # Если на текущей итерации не случилось ни одного обмена,
        # значит массив уже отсортирован, и можем выйти из цикла
        if not swapped:
            break

# Тестирование функции
array = [64, 34, 25, 12, 22, 11, 90]
print("Исходный массив:", array)

bubble_sort(array)

print("Отсортированный массив:", array)

Исходный массив: [64, 34, 25, 12, 22, 11, 90]
Отсортированный массив: [11, 12, 22, 25, 34, 64, 90]

СОРТИРОВКА ВСТАВКАМИ
def insertion_sort(arr):
    
    # Проходим по всему массиву, начиная со второго элемента
    for i in range(1, len(arr)):
        key = arr[i]                     # Текущий элемент, который будем вставлять на нужное место
        j = i - 1                        # Индекс элемента слева от current_element
        
        # Сравниваем ключ с предыдущими элементами, двигаясь назад
        while j >= 0 and key < arr[j]:   # Пока не дошли до начала и элемент слева больше ключа
            arr[j + 1] = arr[j]          # Смещаем больший элемент вправо
            j -= 1                       # Переходим к следующему левому элементу
        
        # Ставим текущий элемент на свободное место
        arr[j + 1] = key                 # Ключ занимает правильное положение

# Пример использования
array = [12, 11, 13, 5, 6]
print("Исходный массив:", array)
insertion_sort(array)
print("Отсортированный массив:", array)

Исходный массив: [12, 11, 13, 5, 6]
Отсортированный массив: [5, 6, 11, 12, 13]

# СОРТИРОВКА ШЕЛЛА
def shell_sort(arr):
    """
    Сортировка Шелла — улучшенная версия сортировки вставками с применением изменяющегося интервала (шага).
    """
    n = len(arr)
    gap = n // 2

    while gap > 0:
        for i in range(gap, n):
            temp = arr[i]
            j = i
            while j >= gap and arr[j - gap] > temp:
                arr[j] = arr[j - gap]
                j -= gap
            arr[j] = temp
        gap //= 2

# Пример
arr = [64, 34, 25, 12, 22, 11, 90]
print("До сортировки:", arr)
shell_sort(arr)
print("После сортировки:", arr)
Пример вывода:
До сортировки:
После сортировки:

python
# БЫСТРАЯ СОРТИРОВКА
def quick_sort(arr):
    """
    Быстрая сортировка (Quicksort).
    Делит массив на три части — меньшие, равные и большие — и рекурсивно сортирует каждую.
    """
    if len(arr) <= 1:
        return arr

    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)

# Проверка
arr = [64, 34, 25, 12, 22, 11, 90]
print("До сортировки:", arr)
result = quick_sort(arr)
print("После сортировки:", result)
Пример вывода:
До сортировки:
После сортировки:

python
# ПОСЛЕДОВАТЕЛЬНЫЙ ПОИСК
def linear_search(arr, target):
    """
    Простая реализация последовательного поиска.
    Проверяет каждый элемент массива, пока не найдёт нужный.
    """
    for i, val in enumerate(arr):
        if val == target:
            return i
    return None

# Пример использования
array = [10, 20, 30, 40, 50, 60, 70]
find = 50
res = linear_search(array, find)

if res is not None:
    print(f"Элемент {find} найден на позиции {res}")
else:
    print(f"Элемент {find} отсутствует")
Пример вывода:
Элемент 50 найден на позиции 4

python
# БИНАРНЫЙ ПОИСК
def binary_search(arr, target):
    """
    Алгоритм бинарного поиска в отсортированном списке.
    """
    low, high = 0, len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        val = arr[mid]

        if val == target:
            return mid
        elif val > target:
            high = mid - 1
        else:
            low = mid + 1

    return None

# Проверка
nums = [10, 20, 30, 40, 50, 60, 70, 80, 90]
x = 50
pos = binary_search(nums, x)

if pos is not None:
    print(f"Элемент {x} найден на позиции {pos}")
else:
    print(f"Элемент {x} не найден")
Пример вывода:
Элемент 50 найден на позиции 4

python
# ПОИСК С ИСПОЛЬЗОВАНИЕМ ЧИСЕЛ ФИБОНАЧЧИ
def fibonacci_search(arr, target):
    """
    Поиск элемента в отсортированном массиве методом Фибоначчи.
    """
    fibMMm2, fibMMm1 = 0, 1
    fibM = fibMMm2 + fibMMm1

    while fibM < len(arr):
        fibMMm2, fibMMm1 = fibMMm1, fibM
        fibM = fibMMm1 + fibMMm2

    offset = -1

    while fibM > 1:
        i = min(offset + fibMMm2, len(arr) - 1)

        if arr[i] < target:
            fibM = fibMMm1
            fibMMm1 = fibMMm2
            fibMMm2 = fibM - fibMMm1
            offset = i
        elif arr[i] > target:
            fibM = fibMMm2
            fibMMm1 -= fibMMm2
            fibMMm2 = fibM - fibMMm1
        else:
            return i

    if fibMMm1 and offset + 1 < len(arr) and arr[offset + 1] == target:
        return offset + 1

    return -1

# Пример
arr = [10, 22, 35, 40, 45, 50, 80, 82, 85, 90, 100]
target = 85
res = fibonacci_search(arr, target)

if res != -1:
    print(f"Элемент {target} найден на позиции {res}")
else:
    print(f"Элемент {target} не найден")
Пример вывода:
Элемент 85 найден на позиции 8