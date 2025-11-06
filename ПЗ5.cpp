#include <iostream>
#include <vector>

/**
 * Рекурсивно проверяет, отсортирован ли массив по возрастанию.
 * 
 * @param arr Вектор чисел для проверки
 * @param index Текущий индекс для сравнения (по умолчанию 0)
 * @return true, если массив отсортирован; false — иначе
 */
bool is_sorted(const std::vector<int>& arr, int index = 0) {
    // Базовый случай: достигли конца массива или массив слишком короткий
    if (index >= static_cast<int>(arr.size()) - 1) {
        return true;
    }
    
    // Если текущая пара в порядке — продолжаем рекурсию
    if (arr[index] <= arr[index + 1]) {
        return is_sorted(arr, index + 1);
    } else {
        // Нашли нарушение порядка
        return false;
    }
}

int main() {
    // Тестовые случаи
    std::vector<std::vector<int>> test_cases = {
        {},                    // пустой массив
        {5},                 // один элемент
        {1, 2, 3, 4},    // отсортированный
        {1, 3, 2, 4},    // не отсортированный
        {2, 2, 3, 4},    // с равными элементами
        {5, 4, 3, 2},    // убывающий порядок
        {1, 1, 1, 1}     // все элементы равны
    };

    for (const auto& arr : test_cases) {
        bool result = is_sorted(arr);
        
        // Вывод массива
        std::cout << "Массив {";
        for (size_t i = 0; i < arr.size(); ++i) {
            std::cout << arr[i];
            if (i < arr.size() - 1) std::cout << ", ";
        }
        std::cout << "} -> " << (result ? "true" : "false") << std::endl;
    }

    return 0;
}











