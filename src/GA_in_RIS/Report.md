# Постановка задачи и аналитика

Для падения поле рассеяния метаповерхности может быть представлено в общем в виде следующим образом 

$$
E(\theta, \varphi) = \sum_{m=0}^{M-1} \sum_{n=1}^{N-1} A_{mn} e^{j\alpha_{mn}} f_{mn}(\theta, \varphi) B_{mn} e^{j\beta_{mn}}
e^{jk_0(md_x\sin\theta \cos \varphi + n d_y \sin \theta \sin \varphi)}
$$

$A_{mn}, \alpha_{mn}$ - амплитуда и фаза падающего поля 

$B_{mn}, \beta_{mn}$ - амплитуда и фаза отраженного поля 

$f_{mn}(\theta, \varphi)$ - ДН ячейки 

$d_x, d_y$ - периодичность в направлениях $x$ и $y$

Для микрополосковой патч-ячейки предполагаем $f_{mn}(\theta, \varphi) = \cos \theta$

$$
E(\theta, \varphi) = \cos \theta \sum_{m=0}^{M-1} \sum_{n=1}^{N-1} A_{mn} e^{j\alpha_{mn}} B_{mn} e^{j\beta_{mn}}
e^{jk_0(md_x\sin\theta \cos \varphi + n d_y \sin \theta \sin \varphi)}
$$

Для простоты считаем, $A_{mn}=B_{mn}=1$ и $\alpha_{mn}=0$

В нашем случае, фаза принимает два значения, так что мы можем закодировать $\varphi_{mn}$ как '1' и '0'. 
Тогда бинарная кодирующая матрица будет иметь вид 

$$
\Phi = 
\begin{pmatrix}
1/0 & 1/0 & \cdots & 1/0 \\ 
1/0 & 1/0 & \cdots & 1/0 \\ 
\vdots & \vdots & \ddots & \vdots \\ 
1/0 & 1/0 & \cdots & 1/0 
\end{pmatrix}
$$

Для оптимизации такого вида матрица, очень удобно ложится Генетический алгоритм 

Определим, критерий по которому будет оценивать распределение 

Интенсивность поля в направлении $(\theta, \varphi)$

$$
I(\theta, \varphi) = |E(\theta, \varphi)|^2 
$$

Будем требовать максимизировать мощность в направлениях всех приемников: 

$$
\max_{\varphi_{mn}} \sum_{k=1}^K w_k I(\theta_r^{(k)}, \varphi_r^{(k)})
$$

где $w_k$ - вес $k$-го направления 

# Генетический алгоритм

Простой генетический алгоритм состоит из трех операций 

1. Репродукция 
2. Кроссинговер 
3. Мутация 

## Репродукция 

Репродукция ist процесс выбора наиболее приспособленных особей из текущей популяции для создания 
следующего поколения на основе значения их целевой функции. Точнее особи с более высоким значением имеют 
более высокую вероятность дать одни или несколько потомков в следующем поколении 

```python
def selection(self, 
              population: np.ndarray,
              fitness_values: np.ndarray) -> np.ndarray:
    total_fitness = np.sum(fitness_values)
    probabilities = fitness_values / total_fitness
    
    parents_indices = np.random.choice(
        len(population), 
        size=len(population), 
        p=probabilities,  
        replace=True
    )
    
    parents = population[parents_indices]
    return parents
```

## Кроссинговер 

Простой кроссинговер реализуется в два этапа

1 Этап: Члены родителей случайным образом образуют пары 

2 Этап: Каждая пара строк подвергается кроссинговеру следующим образом: целая позиция $k$ вдоль строки 
выбирается равномерно случайным образом между 1 и длиной строки. Две новые строки $A_1$ и $A_2$ создаются
путем обмена всеми символами между позициями $k+1$ и длиной строки 

```python
def crossover(self, population):
    new_population = population.copy()
    total_cells = self.M * self.N
    
    indices = list(range(self.population_size))
    
    pairs = []
    for i in range(0, len(indices) - 1, 2):
        pairs.append((indices[i], indices[i + 1]))

    for idx1, idx2 in pairs:
        if np.random.random() < self.pc:
            crossover_point = np.random.randint(1, total_cells)

            p1_flat = new_population[idx1].flatten()
            p2_flat = new_population[idx2].flatten()

            child1_flat = np.concatenate([
                p1_flat[:crossover_point],
                p2_flat[crossover_point:]
            ])

            child2_flat = np.concatenate([
                p2_flat[:crossover_point],
                p1_flat[crossover_point:]
            ])
            
            new_population[idx1] = child1_flat.reshape((self.M, self.N))
            new_population[idx2] = child2_flat.reshape((self.M, self.N))

    return new_population
```

## Мутация 

В простом ГА мутация - это случайное (с небольшой вероятностью) изменение значения позиции строки 

```python
def mutation(self, population):
    new_population = population.copy()
    total_cells = self.M * self.N

    for i in range(self.population_size):
        for j in range(total_cells):
            if np.random.random() < self.pm:
                row = j // self.N
                col = j % self.N

                new_population[i, row, col] = 1 - new_population[i, row, col]

    return new_population
```

## Итоговая оптимизация 

Генетический алгоритм — это итерационный процесс, который повторяет три основные операции 
(репродукция, кроссинговер, мутация) до достижения критерия остановки. В нашей реализации 
критерием остановки является достижение максимального числа поколений max_generations.