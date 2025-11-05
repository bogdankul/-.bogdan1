#Cоздания мультисписка (вложенного списка):
groups = [['House', 'Grass'], ['Cow', 'Milk'], ['Lake', 'Water']]


#Создания очереди:
from queue import Queue
q = Queue()
q.put(1) 
q.put(2) 
q.put(3)


#Реализации дека:
from collections import deque
tasks = deque() 
tasks.append("task1")
tasks.append("task2")
tasks.append("task3") 

#Реализации приоритетной очереди:
from queue import PriorityQueue 
q = PriorityQueue()
q.put((2, 'mid-priority item')) 
q.put((1, 'high-priority item')) 
q.put((3, 'low-priority item')) 


#Пример приоритетной очереди с использованием (Бинарной кучи):
import heapq   
heapq.heappush(customers, (2, "Maks")) 
heapq.heappush(customers, (3, "Magon")) 
heapq.heappush(customers, (1, "Don")) 
heapq.heappush(customers, (4, "Pol")) 