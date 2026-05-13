# Chuẩn hóa Batch
<a id="sec_batch_norm"></a>

Việc huấn luyện mạng nơ-ron sâu là khó khăn.
Làm cho chúng hội tụ trong một khoảng thời gian hợp lý có thể là thách thức.
Trong phần này, chúng ta mô tả *chuẩn hóa batch*, một kỹ thuật phổ biến và hiệu quả
liên tục tăng tốc sự hội tụ của các mạng sâu [Ioffe.Szegedy.2015].
Cùng với các khối dư---được trình bày sau trong [sec_resnet](#sec_resnet)---chuẩn hóa batch
đã giúp các nhà thực hành thường xuyên huấn luyện các mạng với hơn 100 lớp.
Một lợi ích phụ (vô tình) của chuẩn hóa batch nằm ở khả năng chuẩn hóa vốn có của nó.


```python
from d2l import torch as d2l
import torch
from torch import nn
```


## Huấn luyện Mạng Sâu

Khi làm việc với dữ liệu, chúng ta thường tiền xử lý trước khi huấn luyện.
Các lựa chọn liên quan đến tiền xử lý dữ liệu thường tạo ra sự khác biệt lớn trong kết quả cuối cùng.
Nhớ lại ứng dụng MLP của chúng ta để dự đoán giá nhà ([sec_kaggle_house](#sec_kaggle_house)).
Bước đầu tiên của chúng ta khi làm việc với dữ liệu thực
là chuẩn hóa các đặc trưng đầu vào để có
trung bình bằng không $\boldsymbol{\mu} = 0$ và phương sai đơn vị $\boldsymbol{\Sigma} = \boldsymbol{1}$ trên nhiều quan sát [friedman1987exploratory], thường co dãn sao cho đường chéo là một, tức là $\Sigma_{ii} = 1$.
Một chiến lược khác là co dãn các vector về độ dài đơn vị, có thể trung bình bằng không *trên mỗi quan sát*.
Điều này có thể hoạt động tốt, ví dụ, với dữ liệu cảm biến không gian. Các kỹ thuật tiền xử lý này và nhiều kỹ thuật khác,
giúp giữ bài toán ước lượng được kiểm soát tốt.
Để xem xét về lựa chọn và trích xuất đặc trưng, xem bài viết của guyon2008feature, ví dụ.
Chuẩn hóa vector cũng có tác dụng phụ hữu ích là ràng buộc độ phức tạp hàm của các hàm tác động lên nó. Ví dụ, cận radius-margin nổi tiếng [Vapnik95] trong SVM và Định lý Hội tụ Perceptron [Novikoff62] dựa trên đầu vào có chuẩn bị giới hạn.

Theo trực giác, việc chuẩn hóa này phối hợp tốt với các bộ tối ưu hóa của chúng ta
vì nó đặt các tham số *tiên nghiệm* trên một thang đo tương tự.
Vì vậy, thật tự nhiên khi hỏi liệu một bước chuẩn hóa tương ứng *bên trong* mạng sâu
có thể không có lợi hay không. Mặc dù đây không hoàn toàn là lý do dẫn đến sự phát minh ra chuẩn hóa batch [Ioffe.Szegedy.2015], nhưng đây là một cách hữu ích để hiểu nó và họ hàng của nó, chuẩn hóa lớp [Ba.Kiros.Hinton.2016], trong một framework thống nhất.

Thứ hai, đối với một MLP hoặc CNN điển hình, khi chúng ta huấn luyện,
các biến
trong các lớp trung gian (ví dụ: đầu ra biến đổi affine trong MLP)
có thể nhận các giá trị với độ lớn thay đổi rất nhiều:
dù là dọc theo các lớp từ đầu vào đến đầu ra, trên các đơn vị trong cùng lớp,
và theo thời gian do các cập nhật của chúng ta đối với các tham số mô hình.
Những người phát minh ra chuẩn hóa batch đã không chính thức đặt ra giả thuyết
rằng sự trôi dạt trong phân phối của các biến như vậy có thể cản trở sự hội tụ của mạng.
Theo trực giác, chúng ta có thể phỏng đoán rằng nếu một
lớp có các kích hoạt biến thiên gấp 100 lần so với lớp khác,
điều này có thể đòi hỏi các điều chỉnh bù trừ trong tốc độ học. Các bộ giải thích ứng
như AdaGrad [Duchi.Hazan.Singer.2011], Adam [Kingma.Ba.2014], Yogi [Zaheer.Reddi.Sachan.ea.2018], hoặc Distributed Shampoo [anil2020scalable] nhằm giải quyết vấn đề này từ góc độ tối ưu hóa, ví dụ, bằng cách thêm các khía cạnh của phương pháp bậc hai.
Giải pháp thay thế là ngăn vấn đề xảy ra, đơn giản bằng chuẩn hóa thích ứng.

Thứ ba, các mạng sâu hơn phức tạp và có xu hướng dễ bị quá khớp hơn.
Điều này có nghĩa là chuẩn hóa trở nên quan trọng hơn. Một kỹ thuật phổ biến để chuẩn hóa là
tiêm nhiễu. Điều này đã được biết từ lâu, ví dụ, liên quan đến tiêm nhiễu cho
đầu vào [Bishop.1995]. Nó cũng tạo nên cơ sở của dropout trong [sec_dropout](#sec_dropout). Hóa ra, khá vô tình, chuẩn hóa batch mang lại cả ba lợi ích: tiền xử lý, ổn định số học, và chuẩn hóa.

Chuẩn hóa batch được áp dụng cho từng lớp, hoặc tùy chọn, cho tất cả chúng:
Trong mỗi lần lặp huấn luyện,
chúng ta trước tiên chuẩn hóa các đầu vào (của chuẩn hóa batch)
bằng cách trừ trung bình của chúng và
chia cho độ lệch chuẩn của chúng,
trong đó cả hai được ước lượng dựa trên thống kê của minibatch hiện tại.
Tiếp theo, chúng ta áp dụng một hệ số tỷ lệ và một độ dịch để phục hồi các bậc tự do đã mất. Chính do *chuẩn hóa* dựa trên thống kê *batch* này mà *chuẩn hóa batch* có tên gọi như vậy.

Lưu ý rằng nếu chúng ta cố áp dụng chuẩn hóa batch với minibatch kích thước 1,
chúng ta sẽ không thể học được gì.
Đó là vì sau khi trừ trung bình,
mỗi đơn vị ẩn sẽ nhận giá trị 0.
Như bạn có thể đoán, vì chúng ta dành cả một phần cho chuẩn hóa batch,
với minibatch đủ lớn, cách tiếp cận này tỏ ra hiệu quả và ổn định.
Bài học rút ra ở đây là khi áp dụng chuẩn hóa batch,
lựa chọn kích thước batch thậm chí còn quan trọng hơn khi không có chuẩn hóa batch, hoặc ít nhất,
cần hiệu chỉnh phù hợp khi chúng ta có thể điều chỉnh kích thước batch.

Ký hiệu $\mathcal{B}$ là một minibatch và $\mathbf{x} \in \mathcal{B}$ là đầu vào của
chuẩn hóa batch ($\textrm{BN}$). Trong trường hợp này, chuẩn hóa batch được định nghĩa như sau:

$$\textrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$

Trong :eqref:`eq_batchnorm`,
$\hat{\boldsymbol{\mu}}_\mathcal{B}$ là trung bình mẫu
và $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ là độ lệch chuẩn mẫu của minibatch $\mathcal{B}$.
Sau khi áp dụng chuẩn hóa,
minibatch kết quả
có trung bình bằng không và phương sai đơn vị.
Lựa chọn phương sai đơn vị
(thay vì một số ma thuật nào khác) là tùy ý. Chúng ta phục hồi bậc tự do này
bằng cách bao gồm một
*tham số tỷ lệ* theo từng phần tử $\boldsymbol{\gamma}$ và *tham số dịch chuyển* $\boldsymbol{\beta}$
có cùng hình dạng với $\mathbf{x}$. Cả hai đều là các tham số
cần được học như một phần của quá trình huấn luyện mô hình.

Độ lớn biến
cho các lớp trung gian không thể phân kỳ trong quá trình huấn luyện
vì chuẩn hóa batch chủ động căn giữa và co dãn lại chúng
về một trung bình và kích thước nhất định (thông qua $\hat{\boldsymbol{\mu}}_\mathcal{B}$ và ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$).
Kinh nghiệm thực tế xác nhận rằng, như đã gợi ý khi thảo luận về co dãn đặc trưng, chuẩn hóa batch dường như cho phép tốc độ học tập tích cực hơn.
Chúng ta tính $\hat{\boldsymbol{\mu}}_\mathcal{B}$ và ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ trong :eqref:`eq_batchnorm` như sau:

$$\hat{\boldsymbol{\mu}}_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\textrm{ và }
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.$$

Lưu ý rằng chúng ta thêm một hằng số nhỏ $\epsilon > 0$
vào ước lượng phương sai
để đảm bảo rằng chúng ta không bao giờ cố chia cho không,
ngay cả trong trường hợp ước lượng phương sai thực nghiệm có thể rất nhỏ hoặc biến mất.
Các ước lượng $\hat{\boldsymbol{\mu}}_\mathcal{B}$ và ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ chống lại vấn đề tỷ lệ
bằng cách sử dụng các ước lượng nhiễu của trung bình và phương sai.
Bạn có thể nghĩ rằng sự nhiễu loạn này nên là một vấn đề.
Ngược lại, nó thực sự có lợi.

Đây hóa ra là một chủ đề lặp đi lặp lại trong deep learning.
Vì những lý do chưa được đặc trưng hóa tốt về mặt lý thuyết,
nhiều nguồn nhiễu khác nhau trong tối ưu hóa
thường dẫn đến huấn luyện nhanh hơn và ít quá khớp hơn:
sự biến đổi này dường như hoạt động như một dạng chuẩn hóa.
Teye.Azizpour.Smith.2018 và Luo.Wang.Shao.ea.2018
liên kết các tính chất của chuẩn hóa batch với prior Bayesian và các hình phạt tương ứng.
Đặc biệt, điều này làm sáng tỏ một phần câu đố
về lý do tại sao chuẩn hóa batch hoạt động tốt nhất cho các minibatch kích thước vừa phải trong phạm vi 50--100.
Kích thước minibatch cụ thể này dường như tiêm lượng nhiễu "đúng" mỗi lớp, cả về tỷ lệ qua $\hat{\boldsymbol{\sigma}}$, và về độ dịch chuyển qua $\hat{\boldsymbol{\mu}}$: một
minibatch lớn hơn chuẩn hóa ít hơn do các ước lượng ổn định hơn, trong khi các minibatch rất nhỏ
phá hủy tín hiệu hữu ích do phương sai cao. Khám phá hướng này xa hơn, xem xét các loại tiền xử lý và lọc thay thế có thể dẫn đến các loại chuẩn hóa hiệu quả khác.

Khi sửa một mô hình đã được huấn luyện, bạn có thể nghĩ
rằng chúng ta muốn sử dụng toàn bộ tập dữ liệu
để ước lượng trung bình và phương sai.
Sau khi huấn luyện xong, tại sao chúng ta muốn
cùng một ảnh được phân loại khác nhau,
tùy thuộc vào batch mà nó tình cờ nằm trong đó?
Trong quá trình huấn luyện, tính toán chính xác như vậy là không khả thi
vì các biến trung gian
cho tất cả các mẫu dữ liệu
thay đổi mỗi khi chúng ta cập nhật mô hình.
Tuy nhiên, khi mô hình được huấn luyện xong,
chúng ta có thể tính trung bình và phương sai
của các biến của mỗi lớp dựa trên toàn bộ tập dữ liệu.
Thực sự đây là thực hành tiêu chuẩn cho
các mô hình sử dụng chuẩn hóa batch;
do đó các lớp chuẩn hóa batch hoạt động khác nhau
trong *chế độ huấn luyện* (chuẩn hóa theo thống kê minibatch)
so với *chế độ dự đoán* (chuẩn hóa theo thống kê tập dữ liệu).
Ở dạng này, chúng gần giống với hành vi của chuẩn hóa dropout trong [sec_dropout](#sec_dropout),
nơi nhiễu chỉ được tiêm trong quá trình huấn luyện.


## Các Lớp Chuẩn hóa Batch

Các triển khai chuẩn hóa batch cho các lớp kết nối đầy đủ
và các lớp tích chập hơi khác nhau.
Một sự khác biệt chính giữa chuẩn hóa batch và các lớp khác
là vì cái trước hoạt động trên toàn bộ minibatch một lúc,
chúng ta không thể chỉ bỏ qua chiều batch
như chúng ta đã làm trước đây khi giới thiệu các lớp khác.

### Các Lớp Kết nối Đầy đủ

Khi áp dụng chuẩn hóa batch cho các lớp kết nối đầy đủ,
Ioffe.Szegedy.2015, trong bài báo gốc của họ đã chèn chuẩn hóa batch sau phép biến đổi affine
và *trước* hàm kích hoạt phi tuyến. Các ứng dụng sau đó đã thử nghiệm với
việc chèn chuẩn hóa batch ngay *sau* các hàm kích hoạt.
Ký hiệu đầu vào của lớp kết nối đầy đủ là $\mathbf{x}$,
phép biến đổi affine
là $\mathbf{W}\mathbf{x} + \mathbf{b}$ (với tham số trọng số $\mathbf{W}$ và tham số hệ số chặn $\mathbf{b}$),
và hàm kích hoạt là $\phi$,
chúng ta có thể biểu thị phép tính của đầu ra lớp kết nối đầy đủ được kích hoạt chuẩn hóa batch $\mathbf{h}$ như sau:

$$\mathbf{h} = \phi(\textrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

Nhớ lại rằng trung bình và phương sai được tính
trên *cùng* minibatch
mà phép biến đổi được áp dụng.

### Các Lớp Tích chập

Tương tự, với các lớp tích chập,
chúng ta có thể áp dụng chuẩn hóa batch sau tích chập
nhưng trước hàm kích hoạt phi tuyến. Sự khác biệt chính so với chuẩn hóa batch
trong các lớp kết nối đầy đủ là chúng ta áp dụng phép toán trên cơ sở từng kênh
*trên tất cả các vị trí*. Điều này tương thích với giả định bất biến dịch chuyển
của chúng ta dẫn đến tích chập: chúng ta giả định rằng vị trí cụ thể của một mẫu
trong ảnh không quan trọng cho mục đích hiểu biết.

Giả sử các minibatch của chúng ta chứa $m$ mẫu
và đối với mỗi kênh,
đầu ra của tích chập có chiều cao $p$ và chiều rộng $q$.
Đối với các lớp tích chập, chúng ta thực hiện mỗi chuẩn hóa batch
trên $m \cdot p \cdot q$ phần tử trên mỗi kênh đầu ra đồng thời.
Do đó, chúng ta thu thập các giá trị trên tất cả các vị trí không gian
khi tính trung bình và phương sai
và do đó
áp dụng cùng trung bình và phương sai
trong một kênh nhất định
để chuẩn hóa giá trị tại mỗi vị trí không gian.
Mỗi kênh có tham số tỷ lệ và dịch chuyển riêng,
cả hai đều là vô hướng.

### Chuẩn hóa Lớp
<a id="subsec_layer-normalization-in-bn"></a>

Lưu ý rằng trong bối cảnh tích chập, chuẩn hóa batch được định nghĩa rõ ràng ngay cả cho
các minibatch kích thước 1: dù sao chúng ta có tất cả các vị trí trên ảnh để tính trung bình. Do đó,
trung bình và phương sai được xác định rõ ràng, ngay cả khi chỉ trong một quan sát. Xem xét này
đã thúc đẩy Ba.Kiros.Hinton.2016 giới thiệu khái niệm *chuẩn hóa lớp*. Nó hoạt động giống như
một chuẩn hóa batch, chỉ khác là nó được áp dụng cho một quan sát tại một thời điểm. Do đó, cả độ dịch chuyển và hệ số tỷ lệ đều là các vô hướng. Đối với một vector $n$ chiều $\mathbf{x}$, chuẩn lớp được cho bởi

$$\mathbf{x} \rightarrow \textrm{LN}(\mathbf{x}) =  \frac{\mathbf{x} - \hat{\mu}}{\hat\sigma},$$

trong đó tỷ lệ và độ dịch chuyển được áp dụng theo từng hệ số
và được cho bởi

$$\hat{\mu} \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n x_i \textrm{ và }
\hat{\sigma}^2 \stackrel{\textrm{def}}{=} \frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu})^2 + \epsilon.$$

Như trước, chúng ta thêm một độ dịch chuyển nhỏ $\epsilon > 0$ để ngăn chia cho không. Một trong những lợi ích lớn của việc sử dụng chuẩn hóa lớp là nó ngăn phân kỳ. Dù sao, bỏ qua $\epsilon$, đầu ra của chuẩn hóa lớp không phụ thuộc vào tỷ lệ. Tức là, chúng ta có $\textrm{LN}(\mathbf{x}) \approx \textrm{LN}(\alpha \mathbf{x})$ cho bất kỳ lựa chọn nào của $\alpha \neq 0$. Đây trở thành đẳng thức khi $|\alpha| \to \infty$ (xấp xỉ gần đúng là do độ dịch chuyển $\epsilon$ cho phương sai).

Một ưu điểm khác của chuẩn hóa lớp là nó không phụ thuộc vào kích thước minibatch. Nó cũng độc lập với việc chúng ta đang ở chế độ huấn luyện hay kiểm tra. Nói cách khác, đây chỉ đơn giản là một phép biến đổi tất định chuẩn hóa các kích hoạt về một tỷ lệ nhất định. Điều này có thể rất có lợi trong việc ngăn phân kỳ trong tối ưu hóa. Chúng ta bỏ qua các chi tiết thêm và khuyến nghị các độc giả quan tâm tham khảo bài báo gốc.

### Chuẩn hóa Batch trong Dự đoán

Như đã đề cập trước đó, chuẩn hóa batch thường hoạt động khác nhau
trong chế độ huấn luyện so với chế độ dự đoán.
Thứ nhất, nhiễu trong trung bình mẫu và phương sai mẫu
phát sinh từ việc ước lượng từng cái trên minibatch
không còn mong muốn khi chúng ta đã huấn luyện mô hình.
Thứ hai, chúng ta có thể không có khả năng
tính thống kê chuẩn hóa theo batch.
Ví dụ,
chúng ta có thể cần áp dụng mô hình để thực hiện một dự đoán tại một thời điểm.

Thông thường, sau khi huấn luyện, chúng ta sử dụng toàn bộ tập dữ liệu
để tính các ước lượng ổn định của thống kê biến
và sau đó cố định chúng tại thời điểm dự đoán.
Do đó, chuẩn hóa batch hoạt động khác nhau trong quá trình huấn luyện so với lúc kiểm tra.
Nhớ lại rằng dropout cũng thể hiện đặc tính này.

## (**Triển khai từ Đầu**)

Để xem chuẩn hóa batch hoạt động trong thực tế như thế nào, chúng ta triển khai từ đầu bên dưới.


```python
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training mode
    if not torch.is_grad_enabled():
        # In prediction mode, use mean and variance obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used 
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data
```


Bây giờ chúng ta có thể [**tạo một lớp `BatchNorm` đúng đắn.**]
Lớp của chúng ta sẽ duy trì các tham số phù hợp
cho tỷ lệ `gamma` và dịch chuyển `beta`,
cả hai sẽ được cập nhật trong quá trình huấn luyện.
Ngoài ra, lớp của chúng ta sẽ duy trì
các trung bình động của trung bình và phương sai
để sử dụng sau này trong quá trình dự đoán mô hình.

Đặt sang một bên các chi tiết thuật toán,
lưu ý mẫu thiết kế cơ bản trong triển khai lớp của chúng ta.
Thường chúng ta định nghĩa toán học trong một hàm riêng, gọi là `batch_norm`.
Sau đó chúng ta tích hợp chức năng này vào một lớp tùy chỉnh,
mã của nó chủ yếu xử lý các vấn đề quản lý sổ sách,
chẳng hạn như di chuyển dữ liệu đến đúng ngữ cảnh thiết bị,
phân bổ và khởi tạo bất kỳ biến nào cần thiết,
theo dõi các trung bình động (ở đây cho trung bình và phương sai), v.v.
Mẫu này cho phép tách biệt rõ ràng toán học khỏi mã boilerplate.
Cũng lưu ý rằng để thuận tiện
chúng ta không lo lắng về việc tự động suy ra hình dạng đầu vào ở đây;
do đó chúng ta cần chỉ định số lượng đặc trưng xuyên suốt.
Đến nay, tất cả các framework deep learning hiện đại đều cung cấp phát hiện tự động kích thước và hình dạng trong
các API chuẩn hóa batch cấp cao (trong thực tế chúng ta sẽ sử dụng cái này thay thế).


```python
class BatchNorm(nn.Module):
    # num_features: the number of outputs for a fully connected layer or the
    # number of output channels for a convolutional layer. num_dims: 2 for a
    # fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and
        # 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If X is not on the main memory, copy moving_mean and moving_var to
        # the device where X is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1)
        return Y
```


Chúng ta đã sử dụng `momentum` để quản lý việc tổng hợp các ước lượng trung bình và phương sai trong quá khứ. Đây là một cách đặt tên không chính xác vì nó không liên quan gì đến thuật ngữ *động lượng* trong tối ưu hóa. Tuy nhiên, đây là tên thường được áp dụng cho thuật ngữ này và để tuân theo quy ước đặt tên API, chúng ta sử dụng cùng tên biến trong mã của mình.

## [**LeNet với Chuẩn hóa Batch**]

Để xem cách áp dụng `BatchNorm` trong ngữ cảnh,
bên dưới chúng ta áp dụng nó cho mô hình LeNet truyền thống ([sec_lenet](#sec_lenet)).
Nhớ lại rằng chuẩn hóa batch được áp dụng
sau các lớp tích chập hoặc các lớp kết nối đầy đủ
nhưng trước các hàm kích hoạt tương ứng.


Như trước, chúng ta sẽ [**huấn luyện mạng trên tập dữ liệu Fashion-MNIST**].
Mã này gần như giống hệt với khi chúng ta huấn luyện LeNet lần đầu tiên.


Hãy [**xem tham số tỷ lệ `gamma`
và tham số dịch chuyển `beta`**] được học
từ lớp chuẩn hóa batch đầu tiên.


```python
model.net[1].gamma.reshape((-1,)), model.net[1].beta.reshape((-1,))
```


## [**Triển khai Súc tích**]

So với lớp `BatchNorm`
mà chúng ta vừa tự định nghĩa,
chúng ta có thể sử dụng lớp `BatchNorm` được định nghĩa trong các API cấp cao từ framework deep learning trực tiếp.
Mã trông gần như giống hệt
với triển khai của chúng ta ở trên, ngoại trừ chúng ta không còn cần cung cấp thêm đối số để lấy đúng các chiều.


Dưới đây, chúng ta [**sử dụng cùng siêu tham số để huấn luyện mô hình.**]
Lưu ý rằng như thường lệ, biến thể API cấp cao chạy nhanh hơn nhiều
vì mã của nó đã được biên dịch sang C++ hoặc CUDA
trong khi triển khai tùy chỉnh của chúng ta phải được thông dịch bởi Python.


## Thảo luận

Theo trực giác, chuẩn hóa batch được cho là
làm cho cảnh quan tối ưu hóa mượt mà hơn.
Tuy nhiên, chúng ta phải cẩn thận để phân biệt giữa
các trực giác suy đoán và những giải thích đúng đắn
cho các hiện tượng chúng ta quan sát khi huấn luyện mô hình sâu.
Nhớ lại rằng chúng ta thậm chí không biết tại sao
các mạng nơ-ron sâu đơn giản hơn (MLP và CNN thông thường)
tổng quát hóa tốt ngay từ đầu.
Ngay cả với dropout và giảm trọng số,
chúng vẫn linh hoạt đến mức khả năng tổng quát hóa cho dữ liệu chưa thấy
có thể cần các đảm bảo tổng quát hóa lý thuyết học tập tinh vi hơn nhiều.

Bài báo gốc đề xuất chuẩn hóa batch [Ioffe.Szegedy.2015], ngoài việc giới thiệu một công cụ mạnh mẽ và hữu ích,
đã đưa ra giải thích tại sao nó hoạt động:
bằng cách giảm *sự dịch chuyển hiệp biến nội bộ*.
Có lẽ bằng *sự dịch chuyển hiệp biến nội bộ*, họ
muốn nói đến điều gì đó giống như trực giác được biểu đạt ở trên---ý
tưởng rằng phân phối các giá trị biến thay đổi
trong suốt quá trình huấn luyện.
Tuy nhiên, có hai vấn đề với giải thích này:
i) Sự trôi dạt này rất khác với *sự dịch chuyển hiệp biến*,
làm cho tên gọi không chính xác. Nếu có gì, nó gần hơn với sự trôi dạt khái niệm.
ii) Giải thích cung cấp một trực giác không được xác định đầy đủ
nhưng để lại câu hỏi *tại sao chính xác kỹ thuật này hoạt động*
là một câu hỏi mở cần giải thích nghiêm ngặt.
Xuyên suốt cuốn sách này, chúng tôi nhằm truyền đạt các trực giác mà các nhà thực hành
sử dụng để hướng dẫn phát triển mạng nơ-ron sâu của họ.
Tuy nhiên, chúng tôi tin rằng điều quan trọng
là phân tách các trực giác hướng dẫn này
với các sự kiện khoa học đã được xác lập.
Cuối cùng, khi bạn thành thạo tài liệu này
và bắt đầu viết các bài báo nghiên cứu của riêng mình,
bạn sẽ muốn phân biệt rõ ràng
giữa các tuyên bố kỹ thuật và phỏng đoán.

Sau thành công của chuẩn hóa batch,
giải thích của nó về *sự dịch chuyển hiệp biến nội bộ*
đã liên tục xuất hiện trong các cuộc tranh luận trong tài liệu kỹ thuật
và diễn ngôn rộng hơn về cách trình bày nghiên cứu machine learning.
Trong một bài phát biểu đáng nhớ khi nhận Giải thưởng Thử nghiệm theo Thời gian
tại hội nghị NeurIPS 2017,
Ali Rahimi đã sử dụng *sự dịch chuyển hiệp biến nội bộ*
như một điểm trọng tâm trong một lập luận so sánh
thực hành deep learning hiện đại với thuật giả kim.
Sau đó, ví dụ đã được xem xét lại chi tiết
trong một bài báo vị trí nêu rõ
các xu hướng đáng lo ngại trong machine learning [Lipton.Steinhardt.2018].
Các tác giả khác
đã đề xuất các giải thích thay thế cho sự thành công của chuẩn hóa batch,
một số [Santurkar.Tsipras.Ilyas.ea.2018]
khẳng định rằng sự thành công của chuẩn hóa batch đến mặc dù thể hiện hành vi
mà ở một số cách trái ngược với những gì được tuyên bố trong bài báo gốc.


Chúng tôi lưu ý rằng *sự dịch chuyển hiệp biến nội bộ*
không đáng bị phê bình hơn bất kỳ
ngàn tuyên bố mơ hồ tương tự nào
được đưa ra hàng năm trong tài liệu machine learning kỹ thuật.
Có lẽ sự cộng hưởng của nó như một điểm trọng tâm của các cuộc tranh luận này
là do khả năng nhận diện rộng rãi của nó đối với đối tượng mục tiêu.
Chuẩn hóa batch đã được chứng minh là một phương pháp không thể thiếu,
được áp dụng trong hầu hết tất cả các bộ phân loại ảnh được triển khai,
giúp bài báo giới thiệu kỹ thuật này
được trích dẫn hàng chục ngàn lần. Tuy nhiên, chúng tôi phỏng đoán rằng các nguyên tắc hướng dẫn
của chuẩn hóa thông qua tiêm nhiễu, tăng tốc thông qua co dãn lại và cuối cùng là tiền xử lý
có thể dẫn đến các phát minh xa hơn về các lớp và kỹ thuật trong tương lai.

Về mặt thực tế hơn, có một số khía cạnh đáng nhớ về chuẩn hóa batch:

* Trong quá trình huấn luyện mô hình, chuẩn hóa batch liên tục điều chỉnh đầu ra trung gian của
  mạng bằng cách sử dụng trung bình và độ lệch chuẩn của minibatch, để
  các giá trị của đầu ra trung gian trong mỗi lớp trong toàn bộ mạng nơ-ron ổn định hơn.
* Chuẩn hóa batch hơi khác nhau cho các lớp kết nối đầy đủ so với các lớp tích chập. Trên thực tế,
  đối với các lớp tích chập, chuẩn hóa lớp đôi khi có thể được sử dụng như một giải pháp thay thế.
* Giống như một lớp dropout, các lớp chuẩn hóa batch có hành vi khác nhau
  trong chế độ huấn luyện so với chế độ dự đoán.
* Chuẩn hóa batch hữu ích cho chuẩn hóa và cải thiện sự hội tụ trong tối ưu hóa. Ngược lại,
  động cơ ban đầu của việc giảm sự dịch chuyển hiệp biến nội bộ dường như không phải là giải thích hợp lệ.
* Đối với các mô hình mạnh mẽ hơn ít nhạy cảm hơn với các nhiễu loạn đầu vào, hãy cân nhắc việc loại bỏ chuẩn hóa batch [wang2022removing].

## Bài tập

1. Chúng ta có nên loại bỏ tham số hệ số chặn khỏi lớp kết nối đầy đủ hoặc lớp tích chập trước chuẩn hóa batch không? Tại sao?
1. So sánh tốc độ học cho LeNet có và không có chuẩn hóa batch.
    1. Vẽ sự tăng của độ chính xác kiểm định.
    1. Tốc độ học có thể lớn đến đâu trước khi tối ưu hóa thất bại trong cả hai trường hợp?
1. Chúng ta có cần chuẩn hóa batch trong mỗi lớp không? Thử nghiệm với nó.
1. Triển khai một phiên bản "lite" của chuẩn hóa batch chỉ loại bỏ trung bình, hoặc thay thế là chỉ loại bỏ phương sai. Nó hoạt động như thế nào?
1. Cố định các tham số `beta` và `gamma`. Quan sát và phân tích kết quả.
1. Bạn có thể thay thế dropout bằng chuẩn hóa batch không? Hành vi thay đổi như thế nào?
1. Ý tưởng nghiên cứu: hãy nghĩ về các phép biến đổi chuẩn hóa khác mà bạn có thể áp dụng:
    1. Bạn có thể áp dụng phép biến đổi tích phân xác suất không?
    1. Bạn có thể sử dụng ước lượng hiệp phương sai hạng đầy đủ không? Tại sao bạn có thể không nên làm vậy?
    1. Bạn có thể sử dụng các biến thể ma trận nhỏ gọn khác (đường chéo theo khối, hạng dịch chuyển thấp, Monarch, v.v.) không?
    1. Liệu nén thưa thớt có đóng vai trò như một bộ chuẩn hóa không?
    1. Có các phép chiếu khác (ví dụ: côn lồi, phép biến đổi đặc trưng theo nhóm đối xứng) mà bạn có thể sử dụng không?


[Thảo luận](https://discuss.d2l.ai/t/84)
