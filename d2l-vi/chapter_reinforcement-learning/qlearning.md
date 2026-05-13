# Q-Learning
<a id="sec_qlearning"></a>

Trong phần trước, chúng ta đã thảo luận thuật toán Value Iteration, vốn yêu cầu truy cập đầy đủ Markov decision process (MDP), ví dụ các hàm chuyển tiếp và phần thưởng. Trong phần này, chúng ta sẽ xem xét Q-Learning [Watkins.Dayan.1992], một thuật toán học hàm giá trị mà không nhất thiết phải biết MDP. Thuật toán này thể hiện ý tưởng trung tâm đằng sau học tăng cường: nó cho phép robot tự thu thập dữ liệu của mình.
<!-- , instead of relying upon the expert. -->

## Thuật toán Q-Learning

Nhắc lại rằng value iteration cho hàm giá trị-hành động trong :ref:`sec_valueiter` tương ứng với cập nhật

$$Q_{k+1}(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \max_{a' \in \mathcal{A}} Q_k (s', a'); \ \textrm{for all } s \in \mathcal{S} \textrm{ and } a \in \mathcal{A}.$$

Như đã thảo luận, việc triển khai thuật toán này yêu cầu biết MDP, cụ thể là hàm chuyển tiếp $P(s' \mid s, a)$. Ý tưởng then chốt đằng sau Q-Learning là thay thế tổng trên mọi $s' \in \mathcal{S}$ trong biểu thức trên bằng một tổng trên các trạng thái mà robot đã thăm. Điều này cho phép chúng ta tránh nhu cầu phải biết hàm chuyển tiếp.

## Bài toán tối ưu hóa bên dưới Q-Learning

Hãy tưởng tượng robot dùng một chính sách $\pi_e(a \mid s)$ để thực hiện hành động. Giống như chương trước, nó thu thập một bộ dữ liệu gồm $n$ quỹ đạo, mỗi quỹ đạo có $T$ bước thời gian $\{ (s_t^i, a_t^i)_{t=0,\ldots,T-1}\}_{i=1,\ldots, n}$. Nhắc lại rằng value iteration thực chất là một tập ràng buộc liên kết giá trị-hành động $Q^*(s, a)$ của các trạng thái và hành động khác nhau với nhau. Chúng ta có thể triển khai một phiên bản xấp xỉ của value iteration bằng dữ liệu mà robot đã thu thập khi dùng $\pi_e$ như sau

$$\hat{Q} = \min_Q \underbrace{\frac{1}{nT} \sum_{i=1}^n \sum_{t=0}^{T-1} (Q(s_t^i, a_t^i) - r(s_t^i, a_t^i) - \gamma \max_{a'} Q(s_{t+1}^i, a'))^2}_{\stackrel{\textrm{def}}{=} \ell(Q)}.$$

Trước tiên, hãy quan sát các điểm giống và khác nhau giữa biểu thức này và value iteration ở trên. Nếu chính sách của robot $\pi_e$ bằng chính sách tối ưu $\pi^*$, và nếu nó thu thập một lượng dữ liệu vô hạn, thì bài toán tối ưu hóa này sẽ giống hệt bài toán tối ưu hóa bên dưới value iteration. Nhưng trong khi value iteration yêu cầu chúng ta biết $P(s' \mid s, a)$, mục tiêu tối ưu hóa lại không có hạng tử này. Chúng ta không gian lận: khi robot dùng chính sách $\pi_e$ để thực hiện hành động $a_t^i$ tại trạng thái $s_t^i$, trạng thái kế tiếp $s_{t+1}^i$ là một mẫu được rút ra từ hàm chuyển tiếp. Vì vậy, mục tiêu tối ưu hóa cũng có quyền truy cập vào hàm chuyển tiếp, nhưng một cách ngầm định thông qua dữ liệu do robot thu thập.

Các biến của bài toán tối ưu hóa là $Q(s, a)$ với mọi $s \in \mathcal{S}$ và $a \in \mathcal{A}$. Chúng ta có thể tối thiểu hóa mục tiêu bằng gradient descent. Với mỗi cặp $(s_t^i, a_t^i)$ trong bộ dữ liệu, chúng ta có thể viết

$$\begin{aligned}Q(s_t^i, a_t^i) &\leftarrow Q(s_t^i, a_t^i) - \alpha \nabla_{Q(s_t^i,a_t^i)} \ell(Q) \\&=(1 - \alpha) Q(s_t^i,a_t^i) - \alpha \Big( r(s_t^i, a_t^i) + \gamma \max_{a'} Q(s_{t+1}^i, a') \Big),\end{aligned}$$

trong đó $\alpha$ là learning rate. Thông thường trong các bài toán thực tế, khi robot đến được vị trí mục tiêu, các quỹ đạo kết thúc. Giá trị của trạng thái kết thúc như vậy là không vì robot không thực hiện thêm hành động nào sau trạng thái này. Chúng ta nên sửa cập nhật để xử lý các trạng thái như vậy thành

$$Q(s_t^i, a_t^i) =(1 - \alpha) Q(s_t^i,a_t^i) - \alpha \Big( r(s_t^i, a_t^i) + \gamma (1 - \mathbb{1}_{s_{t+1}^i \textrm{ is terminal}} )\max_{a'} Q(s_{t+1}^i, a') \Big).$$

trong đó $\mathbb{1}_{s_{t+1}^i \textrm{ is terminal}}$ là một biến chỉ báo bằng một nếu $s_{t+1}^i$ là trạng thái kết thúc và bằng không nếu ngược lại. Giá trị của các bộ trạng thái-hành động $(s, a)$ không thuộc bộ dữ liệu được đặt là $-\infty$. Thuật toán này được gọi là Q-Learning.

Với nghiệm của các cập nhật này $\hat{Q}$, một xấp xỉ của hàm giá trị tối ưu $Q^*$, chúng ta có thể dễ dàng thu được chính sách tất định tối ưu tương ứng với hàm giá trị này bằng

$$\hat{\pi}(s) = \mathrm{argmax}_{a} \hat{Q}(s, a).$$

Có thể có các tình huống trong đó nhiều chính sách tất định tương ứng với cùng một hàm giá trị tối ưu; các trường hợp hòa như vậy có thể được phá vỡ tùy ý vì chúng có cùng hàm giá trị.

## Khám phá trong Q-Learning

Chính sách được robot dùng để thu thập dữ liệu $\pi_e$ là yếu tố then chốt để đảm bảo Q-Learning hoạt động tốt. Rốt cuộc, chúng ta đã thay thế kỳ vọng trên $s'$ dùng hàm chuyển tiếp $P(s' \mid s, a)$ bằng dữ liệu do robot thu thập. Nếu chính sách $\pi_e$ không đi đến các phần đa dạng của không gian trạng thái-hành động, thì dễ hình dung rằng ước lượng $\hat{Q}$ của chúng ta sẽ là một xấp xỉ kém của $Q^*$ tối ưu. Cũng cần lưu ý rằng trong tình huống như vậy, ước lượng của $Q^*$ tại *mọi trạng thái* $s \in \mathcal{S}$ sẽ kém, không chỉ những trạng thái được $\pi_e$ thăm. Điều này là vì mục tiêu Q-Learning (hoặc value iteration) là một ràng buộc liên kết giá trị của tất cả các cặp trạng thái-hành động. Do đó, việc chọn chính sách $\pi_e$ đúng để thu thập dữ liệu là rất quan trọng.

Chúng ta có thể giảm nhẹ mối lo này bằng cách chọn một chính sách hoàn toàn ngẫu nhiên $\pi_e$, lấy mẫu hành động đều ngẫu nhiên từ $\mathcal{A}$. Một chính sách như vậy sẽ thăm mọi trạng thái, nhưng cần một số lượng lớn quỹ đạo trước khi làm được điều đó.

Từ đó chúng ta đi đến ý tưởng then chốt thứ hai trong Q-Learning, đó là exploration. Các triển khai Q-Learning điển hình liên kết ước lượng hiện tại của $Q$ và chính sách $\pi_e$ để đặt

$$\pi_e(a \mid s) = \begin{cases}\mathrm{argmax}_{a'} \hat{Q}(s, a') & \textrm{with prob. } 1-\epsilon \\ \textrm{uniform}(\mathcal{A}) & \textrm{with prob. } \epsilon,\end{cases}$$

trong đó $\epsilon$ được gọi là "tham số exploration" và do người dùng chọn. Chính sách $\pi_e$ được gọi là chính sách exploration. Chính sách $\pi_e$ cụ thể này được gọi là chính sách exploration $\epsilon$-greedy vì nó chọn hành động tối ưu (theo ước lượng hiện tại $\hat{Q}$) với xác suất $1-\epsilon$ nhưng khám phá ngẫu nhiên với phần xác suất còn lại $\epsilon$. Chúng ta cũng có thể dùng chính sách exploration gọi là softmax

$$\pi_e(a \mid s) = \frac{e^{\hat{Q}(s, a)/T}}{\sum_{a'} e^{\hat{Q}(s, a')/T}};$$

trong đó siêu tham số $T$ được gọi là temperature. Một giá trị lớn của $\epsilon$ trong chính sách $\epsilon$-greedy hoạt động tương tự một giá trị lớn của temperature $T$ trong chính sách softmax.

Điều quan trọng cần lưu ý là khi chúng ta chọn một exploration phụ thuộc vào ước lượng hiện tại của hàm giá trị-hành động $\hat{Q}$, chúng ta cần giải lại bài toán tối ưu hóa theo định kỳ. Các triển khai Q-Learning điển hình thực hiện một cập nhật mini-batch bằng một vài cặp trạng thái-hành động trong bộ dữ liệu đã thu thập (thường là các cặp được thu thập từ bước thời gian trước đó của robot) sau mỗi hành động được thực hiện bằng $\pi_e$.

## Tính chất "tự sửa" của Q-Learning

Bộ dữ liệu do robot thu thập trong quá trình Q-Learning lớn dần theo thời gian. Cả chính sách exploration $\pi_e$ và ước lượng $\hat{Q}$ đều tiến hóa khi robot thu thập thêm dữ liệu. Điều này cho ta một trực giác then chốt về lý do Q-Learning hoạt động tốt. Xét một trạng thái $s$: nếu một hành động cụ thể $a$ có giá trị lớn theo ước lượng hiện tại $\hat{Q}(s,a)$, thì cả chính sách exploration $\epsilon$-greedy và softmax đều có xác suất lớn hơn để chọn hành động này. Nếu hành động này thực ra *không* phải hành động lý tưởng, thì các trạng thái tương lai phát sinh từ hành động này sẽ có phần thưởng kém. Cập nhật tiếp theo của mục tiêu Q-Learning do đó sẽ giảm giá trị $\hat{Q}(s,a)$, làm giảm xác suất chọn hành động này vào lần sau robot thăm trạng thái $s$. Các hành động xấu, ví dụ những hành động có giá trị bị đánh giá quá cao trong $\hat{Q}(s,a)$, được robot khám phá nhưng giá trị của chúng được sửa trong cập nhật tiếp theo của mục tiêu Q-Learning. Các hành động tốt, ví dụ những hành động có giá trị $\hat{Q}(s, a)$ lớn, được robot khám phá thường xuyên hơn và nhờ đó được củng cố. Tính chất này có thể được dùng để chứng minh rằng Q-Learning có thể hội tụ về chính sách tối ưu ngay cả khi nó bắt đầu với một chính sách ngẫu nhiên $\pi_e$ [Watkins.Dayan.1992].

Khả năng không chỉ thu thập dữ liệu mới mà còn thu thập đúng loại dữ liệu là đặc trưng trung tâm của các thuật toán học tăng cường, và đây là điều phân biệt chúng với học có giám sát. Q-Learning, khi dùng mạng nơ-ron sâu (mà chúng ta sẽ thấy trong chương DQN sau này), chịu trách nhiệm cho sự trỗi dậy trở lại của học tăng cường [mnih2013playing].

## Triển khai Q-Learning

Bây giờ chúng ta trình bày cách triển khai Q-Learning trên FrozenLake từ [Open AI Gym](https://gym.openai.com). Lưu ý đây là cùng thiết lập mà chúng ta xét trong thí nghiệm :ref:`sec_valueiter`.

```python

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

seed = 0  # Random number generator seed
gamma = 0.95  # Discount factor
num_iters = 256  # Number of iterations
alpha   = 0.9  # Learing rate
epsilon = 0.9  # Epsilon in epsilion gready algorithm
random.seed(seed)  # Set the random seed
np.random.seed(seed)

# Now set up the environment
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

Trong môi trường FrozenLake, robot di chuyển trên một lưới $4 \times 4$ (đây là các trạng thái) với các hành động là "up" ($\uparrow$), "down" ($\rightarrow$), "left" ($\leftarrow$) và "right" ($\rightarrow$). Môi trường chứa một số ô hố (H), ô đóng băng (F), cũng như một ô mục tiêu (G), tất cả đều chưa biết đối với robot. Để giữ bài toán đơn giản, chúng ta giả định robot có các hành động đáng tin cậy, tức $P(s' \mid s, a) = 1$ với mọi $s \in \mathcal{S}, a \in \mathcal{A}$. Nếu robot đến được mục tiêu, lượt thử kết thúc và robot nhận phần thưởng $1$ bất kể hành động; phần thưởng ở mọi trạng thái khác là $0$ cho tất cả hành động. Mục tiêu của robot là học một chính sách đi đến vị trí mục tiêu (G) từ một vị trí bắt đầu cho trước (S) (đây là $s_0$) để tối đa hóa *return*.

Trước tiên, chúng ta triển khai phương pháp $\epsilon$-greedy như sau:

```python

def e_greedy(env, Q, s, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()

    else:
        return np.argmax(Q[s,:])

```

Bây giờ chúng ta đã sẵn sàng triển khai Q-learning:

```python

def q_learning(env_info, gamma, num_iters, alpha, epsilon):
    env_desc = env_info['desc']  # 2D array specifying what each grid item means
    env = env_info['env']  # 2D array specifying what each grid item means
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']

    Q  = np.zeros((num_states, num_actions))
    V  = np.zeros((num_iters + 1, num_states))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        # Reset environment
        state, done = env.reset(), False
        while not done:
            # Select an action for a given state and acts in env based on selected action
            action = e_greedy(env, Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)

            # Q-update:
            y = reward + gamma * np.max(Q[next_state,:])
            Q[state, action] = Q[state, action] + alpha * (y - Q[state, action])

            # Move to the next state
            state = next_state
        # Record max value and max action for visualization purpose only
        for s in range(num_states):
            V[k,s]  = np.max(Q[s,:])
            pi[k,s] = np.argmax(Q[s,:])
    d2l.show_Q_function_progress(env_desc, V[:-1], pi[:-1])

q_learning(env_info=env_info, gamma=gamma, num_iters=num_iters, alpha=alpha, epsilon=epsilon)

```

Kết quả này cho thấy Q-learning có thể tìm nghiệm tối ưu cho bài toán này sau khoảng 250 lần lặp. Tuy nhiên, khi so sánh kết quả này với kết quả của thuật toán Value Iteration (xem :ref:`subsec_valueitercode`), chúng ta có thể thấy thuật toán Value Iteration cần ít lần lặp hơn nhiều để tìm nghiệm tối ưu cho bài toán này. Điều này xảy ra vì thuật toán Value Iteration có quyền truy cập vào MDP đầy đủ, trong khi Q-learning thì không.


## Tóm tắt
Q-learning là một trong những thuật toán học tăng cường nền tảng nhất. Nó đã ở trung tâm của thành công gần đây của học tăng cường, nổi bật nhất là trong việc học chơi trò chơi điện tử [mnih2013playing]. Việc triển khai Q-learning không yêu cầu chúng ta biết đầy đủ Markov decision process (MDP), ví dụ các hàm chuyển tiếp và phần thưởng.

## Bài tập

1. Thử tăng kích thước lưới lên $8 \times 8$. So với lưới $4 \times 4$, cần bao nhiêu lần lặp để tìm hàm giá trị tối ưu?
1. Chạy lại thuật toán Q-learning với $\gamma$ (tức "gamma" trong đoạn code trên) khi nó bằng $0$, $0.5$ và $1$, rồi phân tích kết quả.
1. Chạy lại thuật toán Q-learning với $\epsilon$ (tức "epsilon" trong đoạn code trên) khi nó bằng $0$, $0.5$ và $1$, rồi phân tích kết quả.

[Thảo luận](https://discuss.d2l.ai/t/12103)
