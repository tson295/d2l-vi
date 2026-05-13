# Value Iteration
<a id="sec_valueiter"></a>

Trong phần này, chúng ta sẽ thảo luận cách chọn hành động tốt nhất cho robot tại mỗi trạng thái để tối đa hóa *return* của quỹ đạo. Chúng ta sẽ mô tả một thuật toán gọi là Value Iteration và triển khai nó cho một robot mô phỏng di chuyển trên một hồ đóng băng.

## Chính sách ngẫu nhiên

Một chính sách ngẫu nhiên, ký hiệu là $\pi(a \mid s)$ (gọi tắt là chính sách), là một phân phối có điều kiện trên các hành động $a \in \mathcal{A}$ khi biết trạng thái $s \in \mathcal{S}$, $\pi(a \mid s) \equiv P(a \mid s)$. Ví dụ, nếu robot có bốn hành động $\mathcal{A}=$ {đi sang trái, đi xuống, đi sang phải, đi lên}. Chính sách tại một trạng thái $s \in \mathcal{S}$ cho tập hành động $\mathcal{A}$ như vậy là một phân phối phân loại, trong đó xác suất của bốn hành động có thể là $[0.4, 0.2, 0.1, 0.3]$; tại một trạng thái khác $s' \in \mathcal{S}$, các xác suất $\pi(a \mid s')$ của cùng bốn hành động có thể là $[0.1, 0.1, 0.2, 0.6]$. Lưu ý rằng ta phải có $\sum_a \pi(a \mid s) = 1$ với bất kỳ trạng thái $s$ nào. Chính sách tất định là một trường hợp đặc biệt của chính sách ngẫu nhiên, trong đó phân phối $\pi(a \mid s)$ chỉ gán xác suất khác không cho một hành động cụ thể, ví dụ $[1, 0, 0, 0]$ trong ví dụ bốn hành động của chúng ta.

Để ký hiệu bớt cồng kềnh, chúng ta thường viết $\pi(s)$ là phân phối có điều kiện thay vì $\pi(a \mid s)$.

## Hàm giá trị

Bây giờ hãy tưởng tượng robot bắt đầu ở trạng thái $s_0$ và tại mỗi thời điểm, trước tiên nó lấy mẫu một hành động từ chính sách $a_t \sim \pi(s_t)$ rồi thực hiện hành động này để dẫn đến trạng thái kế tiếp $s_{t+1}$. Quỹ đạo $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$ có thể khác nhau tùy vào hành động cụ thể $a_t$ nào được lấy mẫu ở các thời điểm trung gian. Chúng ta định nghĩa *return* trung bình $R(\tau) = \sum_{t=0}^\infty \gamma^t r(s_t, a_t)$ của tất cả các quỹ đạo như vậy là
$$V^\pi(s_0) = E_{a_t \sim \pi(s_t)} \Big[ R(\tau) \Big] = E_{a_t \sim \pi(s_t)} \Big[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \Big],$$

trong đó $s_{t+1} \sim P(s_{t+1} \mid s_t, a_t)$ là trạng thái kế tiếp của robot và $r(s_t, a_t)$ là phần thưởng tức thời thu được khi thực hiện hành động $a_t$ tại trạng thái $s_t$ ở thời điểm $t$. Đại lượng này được gọi là "hàm giá trị" của chính sách $\pi$. Nói đơn giản, giá trị của một trạng thái $s_0$ đối với một chính sách $\pi$, ký hiệu là $V^\pi(s_0)$, là *return* kỳ vọng đã chiết khấu bởi $\gamma$ mà robot thu được nếu nó bắt đầu tại trạng thái $s_0$ và thực hiện các hành động từ chính sách $\pi$ tại mỗi thời điểm.

Tiếp theo, chúng ta tách quỹ đạo thành hai giai đoạn: (i) giai đoạn đầu tiên tương ứng với $s_0 \to s_1$ sau khi thực hiện hành động $a_0$, và (ii) giai đoạn thứ hai là quỹ đạo $\tau' = (s_1, a_1, r_1, \ldots)$ sau đó. Ý tưởng then chốt đằng sau mọi thuật toán trong học tăng cường là giá trị của trạng thái $s_0$ có thể được viết thành phần thưởng trung bình thu được trong giai đoạn đầu tiên cộng với hàm giá trị được lấy trung bình trên tất cả các trạng thái kế tiếp khả dĩ $s_1$. Điều này khá trực giác và xuất phát từ giả định Markov của chúng ta: return trung bình từ trạng thái hiện tại là tổng của return trung bình từ trạng thái kế tiếp và phần thưởng trung bình khi đi đến trạng thái kế tiếp. Về mặt toán học, chúng ta viết hai giai đoạn là

$$V^\pi(s_0) = r(s_0, a_0) + \gamma\ E_{a_0 \sim \pi(s_0)} \Big[ E_{s_1 \sim P(s_1 \mid s_0, a_0)} \Big[ V^\pi(s_1) \Big] \Big].$$

Phép phân rã này rất mạnh: nó là nền tảng của nguyên lý quy hoạch động, cơ sở của mọi thuật toán học tăng cường. Lưu ý rằng giai đoạn thứ hai có hai kỳ vọng, một trên các lựa chọn hành động $a_0$ được thực hiện trong giai đoạn đầu bằng chính sách ngẫu nhiên và một trên các trạng thái khả dĩ $s_1$ thu được từ hành động đã chọn. Chúng ta có thể viết :eqref:`eq_dynamic_programming` bằng các xác suất chuyển tiếp trong Markov decision process (MDP) như sau

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$

Một điều quan trọng cần chú ý ở đây là đồng nhất thức trên đúng với mọi trạng thái $s \in \mathcal{S}$ vì chúng ta có thể nghĩ đến bất kỳ quỹ đạo nào bắt đầu tại trạng thái đó và tách quỹ đạo thành hai giai đoạn.

## Hàm giá trị-hành động

Trong triển khai, thường hữu ích khi duy trì một đại lượng gọi là hàm "giá trị hành động", một đại lượng có quan hệ chặt chẽ với hàm giá trị. Nó được định nghĩa là *return* trung bình của một quỹ đạo bắt đầu tại $s_0$ nhưng hành động của giai đoạn đầu được cố định là $a_0$

$$Q^\pi(s_0, a_0) = r(s_0, a_0) + E_{a_t \sim \pi(s_t)} \Big[ \sum_{t=1}^\infty \gamma^t r(s_t, a_t) \Big],$$

lưu ý rằng tổng bên trong kỳ vọng bắt đầu từ $t=1,\ldots, \infty$ vì phần thưởng của giai đoạn đầu đã được cố định trong trường hợp này. Chúng ta lại có thể tách quỹ đạo thành hai phần và viết

$$Q^\pi(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s')\ Q^\pi(s', a');\ \textrm{ for all } s \in \mathcal{S}, a \in \mathcal{A}.$$

Phiên bản này tương tự :eqref:`eq_dynamic_programming_val` cho hàm giá trị-hành động.

## Chính sách ngẫu nhiên tối ưu

Cả hàm giá trị và hàm giá trị-hành động đều phụ thuộc vào chính sách mà robot chọn. Tiếp theo, chúng ta sẽ nghĩ đến "chính sách tối ưu" đạt *return* trung bình lớn nhất
$$\pi^* = \underset{\pi}{\mathrm{argmax}} V^\pi(s_0).$$

Trong tất cả các chính sách ngẫu nhiên khả dĩ mà robot có thể thực hiện, chính sách tối ưu $\pi^*$ đạt *return* chiết khấu trung bình lớn nhất cho các quỹ đạo bắt đầu từ trạng thái $s_0$. Hãy ký hiệu hàm giá trị và hàm giá trị-hành động của chính sách tối ưu là $V^* \equiv V^{\pi^*}$ và $Q^* \equiv Q^{\pi^*}$.

Hãy quan sát rằng với một chính sách tất định, tại bất kỳ trạng thái cho trước nào chỉ có một hành động khả dĩ theo chính sách. Điều này cho ta

$$\pi^*(s) = \underset{a \in \mathcal{A}}{\mathrm{argmax}} \Big[ r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a)\ V^*(s') \Big].$$

Một cách ghi nhớ tốt là hành động tối ưu tại trạng thái $s$ (đối với một chính sách tất định) là hành động tối đa hóa tổng của phần thưởng $r(s, a)$ từ giai đoạn đầu và *return* trung bình của các quỹ đạo bắt đầu từ trạng thái kế tiếp $s'$, được lấy trung bình trên tất cả trạng thái kế tiếp khả dĩ $s'$ từ giai đoạn thứ hai.

## Nguyên lý quy hoạch động

Phần phát triển của chúng ta trong mục trước ở :eqref:`eq_dynamic_programming` hoặc :eqref:`eq_dynamic_programming_q` có thể được biến thành một thuật toán để tính hàm giá trị tối ưu $V^*$ hoặc hàm giá trị-hành động $Q^*$ tương ứng. Quan sát rằng
$$ V^*(s) = \sum_{a \in \mathcal{A}} \pi^*(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$

Với một chính sách tối ưu tất định $\pi^*$, vì chỉ có một hành động có thể được thực hiện tại trạng thái $s$, chúng ta cũng có thể viết

$$V^*(s) = \mathrm{argmax}_{a \in \mathcal{A}} \Big\{ r(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \Big\}$$

với mọi trạng thái $s \in \mathcal{S}$. Đồng nhất thức này được gọi là "nguyên lý quy hoạch động" [BellmanDPPaper, BellmanDPBook]. Nó được Richard Bellman xây dựng vào thập niên 1950 và chúng ta có thể nhớ nó là "phần còn lại của một quỹ đạo tối ưu cũng là tối ưu".

## Value Iteration

Chúng ta có thể biến nguyên lý quy hoạch động thành một thuật toán để tìm hàm giá trị tối ưu, gọi là value iteration. Ý tưởng then chốt đằng sau value iteration là xem đồng nhất thức này như một tập các ràng buộc liên kết $V^*(s)$ tại các trạng thái khác nhau $s \in \mathcal{S}$. Chúng ta khởi tạo hàm giá trị bằng một số giá trị tùy ý $V_0(s)$ cho mọi trạng thái $s \in \mathcal{S}$. Tại lần lặp thứ $k^{\textrm{th}}$, thuật toán Value Iteration cập nhật hàm giá trị như sau

$$V_{k+1}(s) = \max_{a \in \mathcal{A}} \Big\{ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V_k(s') \Big\};\ \textrm{for all } s \in \mathcal{S}.$$

Hóa ra khi $k \to \infty$, hàm giá trị được ước lượng bởi thuật toán Value Iteration hội tụ về hàm giá trị tối ưu bất kể khởi tạo $V_0$,
$$V^*(s) = \lim_{k \to \infty} V_k(s);\ \textrm{for all states } s \in \mathcal{S}.$$

Cùng thuật toán Value Iteration cũng có thể được viết tương đương bằng hàm giá trị-hành động như sau
$$Q_{k+1}(s, a) = r(s, a) + \gamma \max_{a' \in \mathcal{A}} \sum_{s' \in \mathcal{S}} P(s' \mid s, a) Q_k (s', a');\ \textrm{ for all } s \in \mathcal{S}, a \in \mathcal{A}.$$

Trong trường hợp này, chúng ta khởi tạo $Q_0(s, a)$ bằng một số giá trị tùy ý cho mọi $s \in \mathcal{S}$ và $a \in \mathcal{A}$. Một lần nữa, ta có $Q^*(s, a) = \lim_{k \to \infty} Q_k(s, a)$ với mọi $s \in \mathcal{S}$ và $a \in \mathcal{A}$.

## Đánh giá chính sách

Value Iteration cho phép chúng ta tính hàm giá trị tối ưu, tức $V^{\pi^*}$ của chính sách tất định tối ưu $\pi^*$. Chúng ta cũng có thể dùng các cập nhật lặp tương tự để tính hàm giá trị gắn với bất kỳ chính sách nào khác, có thể là ngẫu nhiên, $\pi$. Chúng ta lại khởi tạo $V^\pi_0(s)$ bằng một số giá trị tùy ý cho mọi trạng thái $s \in \mathcal{S}$ và tại lần lặp thứ $k^{\textrm{th}}$, thực hiện các cập nhật

$$    V^\pi_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi_k(s') \Big];\ \textrm{for all } s \in \mathcal{S}.$$

Thuật toán này được gọi là policy evaluation và hữu ích để tính hàm giá trị khi đã biết chính sách. Một lần nữa, hóa ra khi $k \to \infty$, các cập nhật này hội tụ về hàm giá trị đúng bất kể khởi tạo $V_0$,

$$V^\pi(s) = \lim_{k \to \infty} V^\pi_k(s);\ \textrm{for all states } s \in \mathcal{S}.$$

Thuật toán để tính hàm giá trị-hành động $Q^\pi(s, a)$ của một chính sách $\pi$ là tương tự.

## Triển khai Value Iteration
<a id="subsec_valueitercode"></a>
Tiếp theo, chúng ta trình bày cách triển khai Value Iteration cho một bài toán điều hướng gọi là FrozenLake từ [Open AI Gym](https://gym.openai.com). Trước tiên, chúng ta cần thiết lập môi trường như trong đoạn code sau.

```python

%matplotlib inline
import numpy as np
import random
from d2l import torch as d2l

seed = 0  # Random number generator seed
gamma = 0.95  # Discount factor
num_iters = 10  # Number of iterations
random.seed(seed)  # Set the random seed to ensure results can be reproduced
np.random.seed(seed)

# Now set up the environment
env_info = d2l.make_env('FrozenLake-v1', seed=seed)
```

Trong môi trường FrozenLake, robot di chuyển trên một lưới $4 \times 4$ (đây là các trạng thái) với các hành động là "up" ($\uparrow$), "down" ($\rightarrow$), "left" ($\leftarrow$) và "right" ($\rightarrow$). Môi trường chứa một số ô hố (H), ô đóng băng (F), cũng như một ô mục tiêu (G), tất cả đều chưa biết đối với robot. Để giữ bài toán đơn giản, chúng ta giả định robot có các hành động đáng tin cậy, tức $P(s' \mid s, a) = 1$ với mọi $s \in \mathcal{S}, a \in \mathcal{A}$. Nếu robot đến được mục tiêu, lượt thử kết thúc và robot nhận phần thưởng $1$ bất kể hành động; phần thưởng ở mọi trạng thái khác là $0$ cho tất cả hành động. Mục tiêu của robot là học một chính sách đi đến vị trí mục tiêu (G) từ một vị trí bắt đầu cho trước (S) (đây là $s_0$) để tối đa hóa *return*.

Hàm sau triển khai Value Iteration, trong đó `env_info` chứa thông tin liên quan đến MDP và môi trường, còn `gamma` là hệ số chiết khấu:

```python

def value_iteration(env_info, gamma, num_iters):
    env_desc = env_info['desc']  # 2D array shows what each item means
    prob_idx = env_info['trans_prob_idx']
    nextstate_idx = env_info['nextstate_idx']
    reward_idx = env_info['reward_idx']
    num_states = env_info['num_states']
    num_actions = env_info['num_actions']
    mdp = env_info['mdp']

    V  = np.zeros((num_iters + 1, num_states))
    Q  = np.zeros((num_iters + 1, num_states, num_actions))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        for s in range(num_states):
            for a in range(num_actions):
                # Calculate \sum_{s'} p(s'\mid s,a) [r + \gamma v_k(s')]
                for pxrds in mdp[(s,a)]:
                    # mdp(s,a): [(p1,next1,r1,d1),(p2,next2,r2,d2),..]
                    pr = pxrds[prob_idx]  # p(s'\mid s,a)
                    nextstate = pxrds[nextstate_idx]  # Next state
                    reward = pxrds[reward_idx]  # Reward
                    Q[k,s,a] += pr * (reward + gamma * V[k - 1, nextstate])
            # Record max value and max action
            V[k,s] = np.max(Q[k,s,:])
            pi[k,s] = np.argmax(Q[k,s,:])
    d2l.show_value_function_progress(env_desc, V[:-1], pi[:-1])

value_iteration(env_info=env_info, gamma=gamma, num_iters=num_iters)
```

Các hình trên cho thấy chính sách (mũi tên biểu thị hành động) và hàm giá trị (sự thay đổi màu sắc cho thấy hàm giá trị thay đổi theo thời gian từ giá trị ban đầu được biểu thị bằng màu tối đến giá trị tối ưu được biểu thị bằng các màu sáng). Như ta thấy, Value Iteration tìm được hàm giá trị tối ưu sau 10 lần lặp và trạng thái mục tiêu (G) có thể đạt được khi bắt đầu từ bất kỳ trạng thái nào miễn là đó không phải ô H. Một khía cạnh thú vị khác của triển khai là ngoài việc tìm hàm giá trị tối ưu, chúng ta cũng tự động tìm được chính sách tối ưu $\pi^*$ tương ứng với hàm giá trị này.


## Tóm tắt
Ý tưởng chính đằng sau thuật toán Value Iteration là dùng nguyên lý quy hoạch động để tìm return trung bình tối ưu thu được từ một trạng thái cho trước. Lưu ý rằng việc triển khai thuật toán Value Iteration yêu cầu chúng ta biết đầy đủ Markov decision process (MDP), ví dụ như các hàm chuyển tiếp và phần thưởng.


## Bài tập

1. Thử tăng kích thước lưới lên $8 \times 8$. So với lưới $4 \times 4$, cần bao nhiêu lần lặp để tìm hàm giá trị tối ưu?
1. Độ phức tạp tính toán của thuật toán Value Iteration là gì?
1. Chạy lại thuật toán Value Iteration với $\gamma$ (tức "gamma" trong đoạn code trên) khi nó bằng $0$, $0.5$ và $1$, rồi phân tích kết quả.
1. Giá trị của $\gamma$ ảnh hưởng như thế nào đến số lần lặp mà Value Iteration cần để hội tụ? Điều gì xảy ra khi $\gamma=1$?

[Thảo luận](https://discuss.d2l.ai/t/12005)
