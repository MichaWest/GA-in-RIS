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

# Ускорение вычислений с помощью БПФ

Прямой расчет поля по формуле двойной суммы требует $O(MN)$ операций для одного направления и $O((MN)^2)$ для полной картины рассеяния, что крайне неэффективно для больших апертур. Для ускорения используем двумерное дискретное преобразование Фурье (2D-БПФ).

Введем переменные:

$$u = \frac{k_0 d_x \sin\theta \cos\varphi}{2\pi}, \quad v = \frac{k_0 d_y \sin\theta \sin\varphi}{2\pi}$$

Тогда выражение для поля принимает вид:

$$E(u, v) = \cos \theta \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} e^{j\varphi_{mn}} e^{j2\pi(mu + nv)}$$

Это классическое двумерное дискретное преобразование Фурье (ДПФ) от матрицы $e^{j\varphi_{mn}}$.

Таким образом, для вычисления всей картины рассеяния достаточно применить 2D-БПФ к матрице фазовых сдвигов:

$$E(u, v) = \cos \theta \cdot \text{FFT2}\{e^{j\varphi_{mn}}\}$$

Это снижает вычислительную сложность с $O((MN)^2)$ до $O(MN \log(MN))$, что критично для эффективной работы генетического алгоритма.

**Важное замечание:** после применения БПФ полученная картина рассеяния определена в пространстве углов $(u, v)$, которые связаны с физическими углами $(\theta, \varphi)$ соотношениями:

$$u = \frac{d_x}{\lambda} \sin\theta \cos\varphi, \quad v = \frac{d_y}{\lambda} \sin\theta \sin\varphi$$

Диапазон $u \in [-0.5, 0.5]$, $v \in [-0.5, 0.5]$ соответствует всей полусфере.

Метод `calculate_far_field_with_source` демонстрирует практическую реализацию этого подхода:

```python
def calculate_far_field_with_source(self,
                                    coding_matrix: np.ndarray,
                                    theta_source: float,
                                    phi_source: float) -> np.ndarray:
    # 1. Преобразование бинарной кодировки в фазу отражения
    phase_reflection = np.where(coding_matrix == 0, np.pi, 0.0)
    
    # 2. Учет фазовой задержки от источника (если есть)
    phase_source = self.k0 * (
            self.m_indices * self.dx * sin_theta_src * cos_phi_src +
            self.n_indices * self.dy * sin_theta_src * sin_phi_src
    )
    
    # 3. Формирование комплексной матрицы отражения
    total_phase = phase_reflection + phase_source
    reflection_matrix = self.amplitude * np.exp(1j * total_phase)
    
    # 4. Вычисление поля рассеяния через 2D-БПФ
    scattering = fftshift(ifft2(ifftshift(reflection_matrix)))
    
    # 5. Учет диаграммы направленности элемента (cos θ)
    theta, phi = self.get_angles()
    cos_theta = np.cos(theta)
    far_field = scattering * cos_theta / denominator
    
    return far_field
```

Метод `get_angles` выполняет обратное преобразование из координат БПФ в физические углы $(\theta, \varphi)$

```python
def get_angles(self) -> Tuple[np.ndarray, np.ndarray]:
    # Индексы БПФ соответствуют пространственным частотам
    kx = np.arange(-M // 2, M // 2)
    ky = np.arange(-N // 2, N // 2)
    Kx, Ky = np.meshgrid(kx, ky)
    
    # Переход к безразмерным переменным u, v
    u = Kx / (self.k0 * self.dx)
    v = Ky / (self.k0 * self.dy)
    
    # Переход к физическим углам
    r = np.sqrt(u ** 2 + v ** 2)
    theta = np.arcsin(r)
    phi = np.arctan2(v, u)
    
    return theta, phi
```

### Генетический алгоритм

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