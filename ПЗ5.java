public class SortedArrayChecker {

    /**
     * Рекурсивно проверяет, отсортирован ли массив по возрастанию (с допусками на равные элементы).
     *
     * @param arr массив целых чисел для проверки
     * @param index текущий индекс для сравнения (по умолчанию 0)
     * @return true, если массив отсортирован; false — иначе
     */
    public static boolean isSorted(int[] arr, int index) {
        // Базовый случай: достигли конца массива или массив слишком короткий
        if (index >= arr.length - 1) {
            return true;
        }
        
        // Если текущая пара в порядке — продолжаем рекурсию
        if (arr[index] <= arr[index + 1]) {
            return isSorted(arr, index + 1);
        } else {
            // Нашли нарушение порядка
            return false;
        }
    }

    // Упрощённый вызов без указания индекса (начинает с 0)
    public static boolean isSorted(int[] arr) {
        return isSorted(arr, 0);
    }

    public static void main(String[] args) {
        // Тестовые случаи
        int[][] testCases = {
            {},                    // пустой массив
            {5},                 // один элемент
            {1, 2, 3, 4},    // отсортированный
            {1, 3, 2, 4},    // не отсортированный
            {2, 2, 3, 4},    // с равными элементами
            {5, 4, 3, 2},    // убывающий порядок
            {1, 1, 1, 1}     // все элементы равны
        };

        for (int[] arr : testCases) {
            boolean result = isSorted(arr);
            
            // Формируем строковое представление массива
            StringBuilder sb = new StringBuilder();
            sb.append("{");
            for (int i = 0; i < arr.length; i++) {
                sb.append(arr[i]);
                if (i < arr.length - 1) {
                    sb.append(", ");
                }
            }
            sb.append("}");
            
            System.out.println("Массив " + sb.toString() + " -> " + result);
        }
    }
}













