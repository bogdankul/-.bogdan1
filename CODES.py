"""
Блочная(корзинная сортировка)
"""
def insertion_sort(bucket):
    """Используется для сортировки отдельного бака."""
    for i in range(1, len(bucket)):
        up = bucket[i]
        j = i - 1
        while j >= 0 and bucket[j] > up:
            bucket[j + 1] = bucket[j]
            j -= 1
        bucket[j + 1] = up
    return bucket
def bucket_sort(arr):
    """Основная функция блочной сортировки."""
    if not arr or len(arr) <= 1:
        return arr
    
    # Определим максимальный и минимальный элементы
    max_value = max(arr)
    min_value = min(arr)
    
    # Число баков и ширина диапазона для каждого бака
    bucket_count = len(arr)
    bucket_size = (max_value - min_value) / bucket_count
    
    # Создание пустых баков
    buckets = [[] for _ in range(bucket_count)]
    
    # Распределение элементов по бакам
    for value in arr:
        # Выбираем правильный бак для каждого элемента
        index = int((value - min_value) // bucket_size)
        if index != bucket_count:
            buckets[index].append(value)
        else:
            buckets[-1].append(value)  # крайний случай, последний элемент попадает в последний бак
    
    # Сортировка каждого бака
    final_result = []
    for bucket in buckets:
        final_result.extend(insertion_sort(bucket))
    
    return final_result
# Тестирование
numbers = [0.897, 0.565, 0.656, 0.1234, 0.665, 0.3434]
sorted_numbers = bucket_sort(numbers)
print(sorted_numbers)
"""
Вывод:
[0.1234, 0.3434, 0.565, 0.656, 0.665, 0.897]
"""
 
"""
Блинная сортировка
"""
def flip(arr, k):
    """Перевернуть первые k элементов массива"""
    left = 0
    right = k - 1
    while left < right:
        # Меняем местами левый и правый элементы
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1
def pancake_sort(arr):
    """Осуществляет блинную сортировку массива"""
    current_size = len(arr)
    
    # Продолжаем сортировку, пока не останется один элемент
    while current_size > 1:
        # Находим индекс наибольшего элемента в текущей несортированной части
        max_index = arr.index(max(arr[:current_size]))
        
        # Если максимум уже на вершине, перевернем всю текущую часть массива
        if max_index != current_size - 1:
            # Переворот верхней части массива, чтобы максимум попал на вершину
            flip(arr, max_index + 1)
            
            # Теперь переворачиваем всю текущую часть массива, чтобы максимум опустился вниз
            flip(arr, current_size)
        
        # Уменьшаем размер несортированной части
        current_size -= 1
    
    return arr
# Пример использования
arr = [3, 6, 2, 4, 5, 1]
pancake_sorted_arr = pancake_sort(arr)
print(pancake_sorted_arr)
"""
Вывод:
[1, 2, 3, 4, 5, 6]
"""
"""
Сортировка бусинами
"""
def bead_sort(arr):
    """Осуществляет сортировку бусинами"""
    # Получаем максимальную высоту столбца
    max_height = max(arr)
    
    # Формируем матрицу бусинок: True означает наличие бусинки
    beads_matrix = [[False]*len(arr) for _ in range(max_height)]
    
    # Заполнение матрицы бусинками
    for col, height in enumerate(arr):
        # Устанавливаем бусинки в соответствующие ячейки
        for row in range(height):
            beads_matrix[row][col] = True
    
    # "Позволяем бусинкам упасть": собираем их по рядам снизу-вверх
    for row in range(max_height):
        count = sum(beads_matrix[row])  # подсчет количества бусинок в ряду
        # расставляем бусинки слева-направо
        beads_matrix[row] = [True]*count + [False]*(len(arr)-count)
    
    # Восстанавливаем отсортированный массив из матрицы
    sorted_arr = []
    for col in range(len(arr)):
        # подсчёт высоты нового столбца
        height = sum([beads_matrix[row][col] for row in range(max_height)])
        sorted_arr.append(height)
    
    return sorted_arr
# Пример использования
arr = [5, 3, 1, 7, 4]
sorted_arr = bead_sort(arr)
print(sorted_arr)
"""
Вывод:
[7, 5, 4, 3, 1]
"""
"""
Поиск скачками
"""
import math
def jump_search(arr, target):
    """Осуществляет поиск скачком в отсортированном массиве"""
    n = len(arr)
    
    # Определяем оптимальный шаг (квадратный корень из длины массива)
    step = int(math.sqrt(n))
    
    # Начальная точка поиска
    prev = 0
    
    # Находим блок, содержащий целевой элемент
    while arr[min(step, n) - 1] < target:
        prev = step
        step += int(math.sqrt(n))
        if prev >= n:
            return None  # Целевой элемент отсутствует
    
    # Линейный поиск в найденном блоке
    while arr[prev] < target:
        prev += 1
        if prev == min(step, n):
            return None  # Целевой элемент отсутствует
    
    # Проверяем, совпадает ли текущий элемент с целью
    if arr[prev] == target:
        return prev  # Возврат индекса найденного элемента
    
    return None  # Целевой элемент отсутствует
# Пример использования
arr = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
target = 55
result = jump_search(arr, target)
print(f'Элемент {target} найден на индексе: {result}')
"""
Вывод:
Элемент 55 найден на индексе: 10
"""

"""
Экспоненциальный поиск (Exponential Search)
"""
def binary_search(arr, low, high, x):
    """Двоичный поиск внутри заданного диапазона"""
    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1  # Элемент не найден
def exponential_search(arr, x):
    """Осуществляет экспоненциальный поиск в отсортированном массиве"""
    n = len(arr)
    
    # Если массив пустой или цель меньше первого элемента
    if n == 0 or x < arr[0]:
        return -1
    
    # Ищем начальную точку для двоичного поиска
    i = 1
    while i < n and arr[i] <= x:
        i *= 2  # Экспоненциально увеличиваем границу поиска
    
    # Границы для двоичного поиска
    low = i // 2
    high = min(i, n - 1)
    
    # Применяем двоичный поиск в ограниченной области
    return binary_search(arr, low, high, x)
# Пример использования
arr = [2, 3, 4, 10, 40, 50, 60, 70, 80, 90, 100]
target = 10
result = exponential_search(arr, target)
if result != -1:
    print(f'Элемент {target} найден на индексе: {result}')
else:
    print('Элемент не найден')
"""
Вывод:
Элемент 10 найден на индексе: 3
"""
