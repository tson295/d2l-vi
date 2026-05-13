# Prior của Gaussian Process

Hiểu Gaussian process (GP) là điều quan trọng để suy luận về việc xây dựng mô hình và khả năng khái quát hóa, cũng như để đạt hiệu năng tiên tiến trong nhiều ứng dụng, bao gồm active learning và tinh chỉnh siêu tham số trong học sâu. GP xuất hiện ở khắp nơi, và việc biết chúng là gì cũng như cách dùng chúng là có lợi cho chúng ta.

Trong phần này, chúng ta giới thiệu các _prior_ Gaussian process trên hàm. Trong notebook tiếp theo, chúng ta sẽ trình bày cách dùng các prior này để thực hiện _posterior inference_ và đưa ra dự đoán. Phần tiếp theo có thể được xem là "GPs in a nutshell", nhanh chóng cung cấp những gì bạn cần để áp dụng Gaussian process trong thực tế.

```python
from d2l import torch as d2l
import numpy as np
from scipy.spatial import distance_matrix

d2l.set_figsize()
```

## Định nghĩa

Một Gaussian process được định nghĩa là _một tập hợp các biến ngẫu nhiên, trong đó bất kỳ số hữu hạn biến nào cũng có một phân phối Gaussian chung_. Nếu một hàm $f(x)$ là một Gaussian process, với _hàm trung bình_ $m(x)$ và _hàm hiệp phương sai_ hoặc _kernel_ $k(x,x')$, $f(x) \sim \mathcal{GP}(m, k)$, thì bất kỳ tập hợp giá trị hàm nào được truy vấn tại bất kỳ tập hợp điểm đầu vào $x$ nào (thời điểm, vị trí không gian, điểm ảnh, v.v.) đều có một phân phối Gaussian đa biến chung với vector trung bình $\mu$ và ma trận hiệp phương sai $K$: $f(x_1),\dots,f(x_n) \sim \mathcal{N}(\mu, K)$, trong đó $\mu_i = E[f(x_i)] = m(x_i)$ và $K_{ij} = \textrm{Cov}(f(x_i),f(x_j)) = k(x_i,x_j)$.

Định nghĩa này có thể có vẻ trừu tượng và khó tiếp cận, nhưng Gaussian process thực ra là các đối tượng rất đơn giản. Bất kỳ hàm nào

$$f(x) = w^{\top} \phi(x) = \langle w, \phi(x) \rangle,$$

với $w$ được rút từ một phân phối Gaussian (chuẩn), và $\phi$ là bất kỳ vector hàm cơ sở nào, ví dụ $\phi(x) = (1, x, x^2, ..., x^d)^{\top}$,
đều là một Gaussian process. Hơn nữa, bất kỳ Gaussian process f(x) nào cũng có thể được biểu diễn dưới dạng của phương trình :eqref:`eq_gp-function`. Hãy xét một vài ví dụ cụ thể để bắt đầu làm quen với Gaussian process, sau đó ta có thể đánh giá chúng thật sự đơn giản và hữu ích đến mức nào.

## Một Gaussian Process đơn giản

Giả sử $f(x) = w_0 + w_1 x$, và $w_0, w_1 \sim \mathcal{N}(0,1)$, với $w_0, w_1, x$ đều một chiều. Chúng ta có thể viết tương đương hàm này dưới dạng tích trong $f(x) = (w_0, w_1)(1, x)^{\top}$. Trong :eqref:`eq_gp-function` ở trên, $w = (w_0, w_1)^{\top}$ và $\phi(x) = (1,x)^{\top}$.

Với bất kỳ $x$ nào, $f(x)$ là tổng của hai biến ngẫu nhiên Gaussian. Vì Gaussian đóng dưới phép cộng, $f(x)$ cũng là một biến ngẫu nhiên Gaussian với bất kỳ $x$ nào. Thực ra, với bất kỳ $x$ cụ thể nào, ta có thể tính được $f(x)$ là $\mathcal{N}(0,1+x^2)$. Tương tự, phân phối chung cho bất kỳ tập hợp giá trị hàm nào, $(f(x_1),\dots,f(x_n))$, với bất kỳ tập hợp đầu vào $x_1,\dots,x_n$ nào, là một phân phối Gaussian đa biến. Do đó $f(x)$ là một Gaussian process.

Tóm lại, $f(x)$ là một _hàm ngẫu nhiên_, hay một _phân phối trên các hàm_. Chúng ta có thể có thêm trực giác về phân phối này bằng cách lặp lại việc lấy mẫu các giá trị cho $w_0, w_1$, và trực quan hóa các hàm $f(x)$ tương ứng, vốn là các đường thẳng với độ dốc và hệ số chặn khác nhau, như sau:

```python
def lin_func(x, n_sample):
    preds = np.zeros((n_sample, x.shape[0]))
    for ii in range(n_sample):
        w = np.random.normal(0, 1, 2)
        y = w[0] + w[1] * x
        preds[ii, :] = y
    return preds

x_points = np.linspace(-5, 5, 50)
outs = lin_func(x_points, 10)
lw_bd = -2 * np.sqrt((1 + x_points ** 2))
up_bd = 2 * np.sqrt((1 + x_points ** 2))

d2l.plt.fill_between(x_points, lw_bd, up_bd, alpha=0.25)
d2l.plt.plot(x_points, np.zeros(len(x_points)), linewidth=4, color='black')
d2l.plt.plot(x_points, outs.T)
d2l.plt.xlabel("x", fontsize=20)
d2l.plt.ylabel("f(x)", fontsize=20)
d2l.plt.show()
```

Nếu thay vào đó $w_0$ và $w_1$ được rút từ $\mathcal{N}(0,\alpha^2)$, bạn hình dung việc thay đổi $\alpha$ ảnh hưởng thế nào đến phân phối trên các hàm?

## Từ không gian trọng số đến không gian hàm

Trong đồ thị ở trên, chúng ta đã thấy cách một phân phối trên tham số trong một mô hình cảm sinh một phân phối trên các hàm. Dù chúng ta thường có ý tưởng về các hàm mình muốn mô hình hóa, chẳng hạn chúng có trơn, tuần hoàn, biến thiên nhanh, v.v. hay không, việc suy luận về các tham số, vốn phần lớn khó diễn giải, lại tương đối tẻ nhạt. May mắn là Gaussian process cung cấp một cơ chế dễ dàng để suy luận _trực tiếp_ về các hàm. Vì một phân phối Gaussian được xác định hoàn toàn bởi hai moment đầu tiên, tức trung bình và ma trận hiệp phương sai, nên theo mở rộng, một Gaussian process được xác định bởi hàm trung bình và hàm hiệp phương sai của nó.

Trong ví dụ trên, hàm trung bình là

$$m(x) = E[f(x)] = E[w_0 + w_1x] = E[w_0] + E[w_1]x = 0+0 = 0.$$

Tương tự, hàm hiệp phương sai là

$$k(x,x') = \textrm{Cov}(f(x),f(x')) = E[f(x)f(x')]-E[f(x)]E[f(x')] = E[w_0^2 + w_0w_1x' + w_1w_0x + w_1^2xx'] = 1 + xx'.$$

Phân phối trên các hàm của chúng ta giờ có thể được chỉ định và lấy mẫu trực tiếp, mà không cần lấy mẫu từ phân phối trên tham số. Ví dụ, để rút mẫu từ $f(x)$, chúng ta chỉ cần lập phân phối Gaussian đa biến gắn với bất kỳ tập $x$ nào muốn truy vấn, rồi lấy mẫu trực tiếp từ đó. Chúng ta sẽ bắt đầu thấy công thức hóa này có lợi đến mức nào.

Trước tiên, ta lưu ý rằng về cơ bản cùng suy luận như với mô hình đường thẳng đơn giản ở trên có thể được áp dụng để tìm hàm trung bình và hàm hiệp phương sai cho _bất kỳ_ mô hình nào có dạng $f(x) = w^{\top} \phi(x)$, với $w \sim \mathcal{N}(u,S)$. Trong trường hợp này, hàm trung bình $m(x) = u^{\top}\phi(x)$, và hàm hiệp phương sai $k(x,x') = \phi(x)^{\top}S\phi(x')$. Vì $\phi(x)$ có thể biểu diễn một vector gồm bất kỳ hàm cơ sở phi tuyến nào, chúng ta đang xét một lớp mô hình rất tổng quát, bao gồm cả các mô hình có _vô hạn_ tham số.

## Kernel Radial Basis Function (RBF)

_Radial basis function_ (RBF) kernel là hàm hiệp phương sai phổ biến nhất cho Gaussian process, và nói chung cho kernel machine.
Kernel này có dạng $k_{\textrm{RBF}}(x,x') = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right)$, trong đó $a$ là tham số biên độ, và $\ell$ là siêu tham số _lengthscale_.

Hãy suy ra kernel này bắt đầu từ không gian trọng số. Xét hàm

$$f(x) = \sum_{i=1}^J w_i \phi_i(x), w_i  \sim \mathcal{N}\left(0,\frac{\sigma^2}{J}\right), \phi_i(x) = \exp\left(-\frac{(x-c_i)^2}{2\ell^2 }\right).$$

$f(x)$ là tổng của các hàm cơ sở xuyên tâm, với độ rộng $\ell$, có tâm tại các điểm $c_i$, như trong hình sau.


Chúng ta có thể nhận ra $f(x)$ có dạng $w^{\top} \phi(x)$, trong đó $w = (w_1,\dots,w_J)^{\top}$ và $\phi(x)$ là một vector chứa từng hàm cơ sở xuyên tâm. Khi đó hàm hiệp phương sai của Gaussian process này là

$$k(x,x') = \frac{\sigma^2}{J} \sum_{i=1}^{J} \phi_i(x)\phi_i(x').$$

Bây giờ hãy xét điều gì xảy ra khi ta đưa số tham số (và hàm cơ sở) đến vô hạn. Đặt $c_J = \log J$, $c_1 = -\log J$, và $c_{i+1}-c_{i} = \Delta c = 2\frac{\log J}{J}$, và $J \to \infty$. Hàm hiệp phương sai trở thành tổng Riemann:

$$k(x,x') = \lim_{J \to \infty} \frac{\sigma^2}{J} \sum_{i=1}^{J} \phi_i(x)\phi_i(x') = \int_{c_0}^{c_\infty} \phi_c(x)\phi_c(x') dc.$$

Bằng cách đặt $c_0 = -\infty$ và $c_\infty = \infty$, chúng ta trải vô hạn hàm cơ sở trên toàn bộ trục thực, mỗi hàm cách nhau
một khoảng $\Delta c \to 0$:

$$k(x,x') = \int_{-\infty}^{\infty} \exp(-\frac{(x-c)^2}{2\ell^2}) \exp(-\frac{(x'-c)^2}{2\ell^2 }) dc = \sqrt{\pi}\ell \sigma^2 \exp(-\frac{(x-x')^2}{2(\sqrt{2} \ell)^2}) \propto k_{\textrm{RBF}}(x,x').$$

Đáng dành một chút thời gian để hấp thụ điều chúng ta vừa làm. Bằng cách chuyển sang biểu diễn không gian hàm, chúng ta đã suy ra cách biểu diễn một mô hình có _vô hạn_ tham số bằng một lượng tính toán hữu hạn. Một Gaussian process với RBF kernel là một _bộ xấp xỉ phổ dụng_, có khả năng biểu diễn bất kỳ hàm liên tục nào với độ chính xác tùy ý. Ta có thể thấy trực giác vì sao từ suy luận ở trên. Chúng ta có thể làm mỗi hàm cơ sở xuyên tâm co lại thành một khối điểm bằng cách lấy $\ell \to 0$, và gán cho mỗi khối điểm bất kỳ chiều cao nào mình muốn.

Vậy một Gaussian process với RBF kernel là một mô hình có vô hạn tham số và linh hoạt hơn nhiều so với bất kỳ mạng nơ-ron hữu hạn nào. Có lẽ những tranh luận về mạng nơ-ron _overparametrized_ đã đặt nhầm trọng tâm. Như chúng ta sẽ thấy, GP với RBF kernel không overfit, và thật ra còn đem lại hiệu năng khái quát hóa đặc biệt thuyết phục trên các bộ dữ liệu nhỏ. Hơn nữa, các ví dụ trong [zhang2021understanding], chẳng hạn khả năng khớp hoàn hảo ảnh với nhãn ngẫu nhiên nhưng vẫn khái quát hóa tốt trên các bài toán có cấu trúc, (có thể được tái tạo hoàn hảo bằng Gaussian process) [wilson2020bayesian]. Mạng nơ-ron không khác biệt như ta thường nghĩ.

Chúng ta có thể xây dựng thêm trực giác về Gaussian process với RBF kernel và các siêu tham số như _length-scale_ bằng cách lấy mẫu trực tiếp từ phân phối trên các hàm. Như trước, việc này gồm một quy trình đơn giản:

1. Chọn các điểm đầu vào $x$ mà ta muốn truy vấn GP: $x_1,\dots,x_n$.
2. Đánh giá $m(x_i)$, $i = 1,\dots,n$, và $k(x_i,x_j)$ với $i,j = 1,\dots,n$ để lần lượt tạo vector trung bình và ma trận hiệp phương sai $\mu$ và $K$, trong đó $(f(x_1),\dots,f(x_n)) \sim \mathcal{N}(\mu, K)$.
3. Lấy mẫu từ phân phối Gaussian đa biến này để thu được các giá trị hàm mẫu.
4. Lấy mẫu thêm nhiều lần để trực quan hóa nhiều hàm mẫu hơn được truy vấn tại các điểm đó.

Chúng ta minh họa quy trình này trong hình bên dưới.

```python
def rbfkernel(x1, x2, ls=4.):  
    dist = distance_matrix(np.expand_dims(x1, 1), np.expand_dims(x2, 1))
    return np.exp(-(1. / ls / 2) * (dist ** 2))

x_points = np.linspace(0, 5, 50)
meanvec = np.zeros(len(x_points))
covmat = rbfkernel(x_points,x_points, 1)

prior_samples= np.random.multivariate_normal(meanvec, covmat, size=5);
d2l.plt.plot(x_points, prior_samples.T, alpha=0.5)
d2l.plt.show()
```

## Kernel mạng nơ-ron

Nghiên cứu về Gaussian process trong học máy được khơi nguồn bởi nghiên cứu về mạng nơ-ron. Radford Neal theo đuổi các mạng nơ-ron Bayes ngày càng lớn, cuối cùng chứng minh vào năm 1994 (sau này xuất bản năm 1996, vì đây là một trong những bài bị từ chối khét tiếng nhất ở NeurIPS) rằng các mạng như vậy với số lượng đơn vị ẩn vô hạn trở thành Gaussian process với các hàm kernel cụ thể [neal1996bayesian]. Sự quan tâm đến suy luận này đã xuất hiện trở lại, với các ý tưởng như neural tangent kernel được dùng để khảo sát các tính chất khái quát hóa của mạng nơ-ron [matthews2018gaussian] [novak2018bayesian]. Chúng ta có thể suy ra kernel mạng nơ-ron như sau.

Xét một hàm mạng nơ-ron $f(x)$ với một tầng ẩn:

$$f(x) = b + \sum_{i=1}^{J} v_i h(x; u_i).$$

$b$ là bias, $v_i$ là trọng số từ ẩn đến đầu ra, $h$ là bất kỳ hàm truyền đơn vị ẩn bị chặn nào, $u_i$ là trọng số từ đầu vào đến ẩn, và $J$ là số đơn vị ẩn. Cho $b$ và $v_i$ độc lập với trung bình không và phương sai lần lượt là $\sigma_b^2$ và $\sigma_v^2/J$, đồng thời cho các $u_i$ có phân phối độc lập đồng nhất. Khi đó chúng ta có thể dùng định lý giới hạn trung tâm để chứng minh rằng bất kỳ tập giá trị hàm nào $f(x_1),\dots,f(x_n)$ cũng có một phân phối Gaussian đa biến chung.

Hàm trung bình và hiệp phương sai của Gaussian process tương ứng là:

$$m(x) = E[f(x)] = 0$$

$$k(x,x') = \textrm{cov}[f(x),f(x')] = E[f(x)f(x')] = \sigma_b^2 + \frac{1}{J} \sum_{i=1}^{J} \sigma_v^2 E[h_i(x; u_i)h_i(x'; u_i)]$$

Trong một số trường hợp, về cơ bản chúng ta có thể đánh giá hàm hiệp phương sai này ở dạng đóng. Cho $h(x; u) = \textrm{erf}(u_0 + \sum_{j=1}^{P} u_j x_j)$, trong đó $\textrm{erf}(z) = \frac{2}{\sqrt{\pi}} \int_{0}^{z} e^{-t^2} dt$, và $u \sim \mathcal{N}(0,\Sigma)$. Khi đó $k(x,x') = \frac{2}{\pi} \textrm{sin}(\frac{2 \tilde{x}^{\top} \Sigma \tilde{x}'}{\sqrt{(1 + 2 \tilde{x}^{\top} \Sigma \tilde{x})(1 + 2 \tilde{x}'^{\top} \Sigma \tilde{x}')}})$.

RBF kernel là _dừng_, nghĩa là nó _bất biến tịnh tiến_, và do đó có thể được viết như một hàm của $\tau = x-x'$. Trực giác mà nói, tính dừng có nghĩa là các tính chất cấp cao của hàm, chẳng hạn tốc độ biến thiên, không thay đổi khi ta di chuyển trong không gian đầu vào. Tuy nhiên, kernel mạng nơ-ron là _không dừng_. Bên dưới, chúng ta hiển thị các hàm mẫu từ một Gaussian process với kernel này. Ta có thể thấy hàm trông khác về chất gần gốc tọa độ.

## Tóm tắt


Bước đầu tiên khi thực hiện suy luận Bayes là chỉ định một prior. Gaussian process có thể được dùng để chỉ định toàn bộ prior trên các hàm. Bắt đầu từ góc nhìn "không gian trọng số" truyền thống của mô hình hóa, chúng ta có thể cảm sinh một prior trên các hàm bằng cách bắt đầu với dạng hàm của một mô hình và đưa vào một phân phối trên các tham số của nó. Thay vào đó, chúng ta cũng có thể chỉ định một phân phối prior trực tiếp trong không gian hàm, với các tính chất được kiểm soát bởi một kernel. Cách tiếp cận không gian hàm có nhiều lợi thế. Chúng ta có thể xây dựng các mô hình thực sự tương ứng với vô hạn tham số, nhưng dùng một lượng tính toán hữu hạn! Hơn nữa, dù các mô hình này có mức linh hoạt rất lớn, chúng cũng đưa ra các giả định mạnh về những loại hàm nào là có khả năng trước khi quan sát dữ liệu, dẫn đến khả năng khái quát hóa tương đối tốt trên các bộ dữ liệu nhỏ.

Các giả định của mô hình trong không gian hàm được kiểm soát trực giác bởi các kernel, thường mã hóa các tính chất cấp cao hơn của hàm, chẳng hạn độ trơn và tính chu kỳ. Nhiều kernel là dừng, nghĩa là chúng bất biến tịnh tiến. Các hàm được rút từ một Gaussian process với kernel dừng có xấp xỉ cùng các tính chất cấp cao (chẳng hạn tốc độ biến thiên) bất kể ta nhìn ở đâu trong không gian đầu vào.

Gaussian process là một lớp mô hình tương đối tổng quát, chứa nhiều ví dụ về các mô hình mà chúng ta đã quen thuộc, bao gồm đa thức, chuỗi Fourier, v.v., miễn là ta có prior Gaussian trên các tham số. Chúng cũng bao gồm các mạng nơ-ron với số lượng tham số vô hạn, ngay cả khi không có phân phối Gaussian trên tham số. Mối liên hệ này, do Radford Neal phát hiện, đã thúc đẩy các nhà nghiên cứu học máy chuyển từ mạng nơ-ron sang Gaussian process.


## Bài tập

1. Vẽ các hàm prior mẫu từ một GP với kernel Ornstein-Uhlenbeck (OU), $k_{\textrm{OU}}(x,x') = \exp\left(-\frac{1}{2\ell}||x - x'|\right)$. Nếu bạn cố định lengthscale $\ell$ giống nhau, các hàm này trông khác các hàm mẫu từ GP với RBF kernel như thế nào?

2. Việc thay đổi _biên độ_ $a^2$ của RBF kernel ảnh hưởng thế nào đến phân phối trên các hàm?

3. Giả sử ta tạo $u(x) = f(x) + 2g(x)$, trong đó $f(x) \sim \mathcal{GP}(m_1,k_1)$ và $g(x) \sim \mathcal{GP}(m_2,k_2)$. $u(x)$ có phải là một Gaussian process không, và nếu có thì hàm trung bình và hàm hiệp phương sai của nó là gì?

4. Giả sử ta tạo $g(x) = a(x)f(x)$, trong đó $f(x) \sim \mathcal{GP}(0,k)$ và $a(x) = x^2$. $g(x)$ có phải là một Gaussian process không, và nếu có thì hàm trung bình và hàm hiệp phương sai của nó là gì? Tác động của $a(x)$ là gì? Các hàm mẫu được rút từ $g(x)$ trông như thế nào?

5. Giả sử ta tạo $u(x) = f(x)g(x)$, trong đó $f(x) \sim \mathcal{GP}(m_1,k_1)$ và $g(x) \sim \mathcal{GP}(m_2,k_2)$. $u(x)$ có phải là một Gaussian process không, và nếu có thì hàm trung bình và hàm hiệp phương sai của nó là gì?

[Thảo luận](https://discuss.d2l.ai/t/12116)
