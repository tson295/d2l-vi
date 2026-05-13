# Thiết kế Kiến trúc Mạng Tích chập
<a id="sec_cnn-design"></a>

Các phần trước đã dẫn chúng ta qua một chuyến tham quan về thiết kế mạng hiện đại cho thị giác máy tính. Điểm chung của tất cả công việc chúng ta đã đề cập là nó phụ thuộc rất nhiều vào trực giác của các nhà khoa học. Nhiều kiến trúc được thông báo nhiều bởi sự sáng tạo của con người và ở mức độ thấp hơn nhiều bởi việc khám phá có hệ thống không gian thiết kế mà các mạng sâu cung cấp. Tuy nhiên, cách tiếp cận *kỹ thuật mạng* này đã cực kỳ thành công.

Kể từ khi AlexNet ([sec_alexnet](#sec_alexnet))
đánh bại các mô hình thị giác máy tính thông thường trên ImageNet,
việc xây dựng các mạng rất sâu
bằng cách xếp chồng các khối tích chập, tất cả được thiết kế theo cùng một mẫu đã trở nên phổ biến.
Đặc biệt, tích chập $3 \times 3$ đã được
phổ biến bởi các mạng VGG ([sec_vgg](#sec_vgg)).
NiN ([sec_nin](#sec_nin)) cho thấy ngay cả tích chập $1 \times 1$ cũng có thể
có lợi bằng cách thêm phi tuyến cục bộ.
Hơn nữa, NiN giải quyết vấn đề tổng hợp thông tin tại đầu mạng
bằng cách tổng hợp trên tất cả các vị trí.
GoogLeNet ([sec_googlenet](#sec_googlenet)) thêm nhiều nhánh tích chập có chiều rộng khác nhau,
kết hợp ưu điểm của VGG và NiN trong khối Inception của nó.
ResNet ([sec_resnet](#sec_resnet))
thay đổi thiên kiến quy nạp về phía ánh xạ đồng nhất (từ $f(x) = 0$). Điều này cho phép có các mạng rất sâu. Gần một thập kỷ sau, thiết kế ResNet vẫn còn phổ biến, là bằng chứng cho thiết kế của nó. Cuối cùng, ResNeXt ([subsec_resnext](#subsec_resnext)) đã thêm các tích chập theo nhóm, cung cấp sự cân bằng tốt hơn giữa tham số và tính toán. Là tiền thân của Transformer cho thị giác, Mạng Squeeze-and-Excitation (SENet) cho phép truyền thông tin hiệu quả giữa các vị trí
[Hu.Shen.Sun.2018]. Điều này được thực hiện bằng cách tính toán một hàm attention toàn cục theo từng kênh.

Cho đến nay chúng ta đã bỏ qua các mạng thu được qua *tìm kiếm kiến trúc nơ-ron* (NAS) [zoph2016neural, liu2018darts]. Chúng ta chọn làm như vậy vì chi phí của chúng thường rất lớn, dựa vào tìm kiếm brute-force, thuật toán di truyền, học tăng cường, hoặc một số dạng tối ưu hóa siêu tham số khác. Cho một không gian tìm kiếm cố định,
NAS sử dụng một chiến lược tìm kiếm để tự động lựa chọn
một kiến trúc dựa trên ước lượng hiệu suất được trả về.
Kết quả của NAS
là một thể hiện mạng duy nhất. EfficientNets là một kết quả đáng chú ý của tìm kiếm này [tan2019efficientnet].

Tiếp theo chúng ta thảo luận về một ý tưởng khá khác với việc tìm kiếm *một mạng tốt nhất duy nhất*. Nó tương đối rẻ về mặt tính toán, nó dẫn đến những hiểu biết khoa học trên đường đi, và nó khá hiệu quả về chất lượng kết quả. Hãy xem xét chiến lược của Radosavovic.Kosaraju.Girshick.ea.2020 để *thiết kế không gian thiết kế mạng*. Chiến lược này kết hợp điểm mạnh của thiết kế thủ công và NAS. Nó thực hiện điều này bằng cách hoạt động trên *phân phối của mạng* và tối ưu hóa phân phối theo cách để thu được hiệu suất tốt cho toàn bộ các họ mạng. Kết quả là các *RegNet*, cụ thể là RegNetX và RegNetY, cộng với một loạt các nguyên tắc hướng dẫn cho thiết kế CNN có hiệu suất tốt.


```python
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```


## Không gian Thiết kế AnyNet
<a id="subsec_the-anynet-design-space"></a>

Mô tả dưới đây tuân theo chặt chẽ lý luận trong Radosavovic.Kosaraju.Girshick.ea.2020 với một số rút gọn để phù hợp với phạm vi của cuốn sách.
Để bắt đầu, chúng ta cần một mẫu cho họ mạng cần khám phá. Một trong những điểm chung của các thiết kế trong chương này là các mạng bao gồm một *thân*, một *thân chính* và một *đầu*. Thân thực hiện xử lý ảnh ban đầu, thường qua các tích chập với kích thước cửa sổ lớn hơn. Thân chính bao gồm nhiều khối, thực hiện phần lớn các phép biến đổi cần thiết để đi từ ảnh thô đến biểu diễn đối tượng. Cuối cùng, đầu chuyển đổi điều này thành các đầu ra mong muốn, chẳng hạn như qua bộ hồi quy softmax cho phân loại đa lớp.
Thân chính, lần lượt, bao gồm nhiều giai đoạn, hoạt động trên ảnh ở độ phân giải giảm dần. Trên thực tế, cả thân và mỗi giai đoạn tiếp theo đều giảm một phần tư độ phân giải không gian. Cuối cùng, mỗi giai đoạn bao gồm một hoặc nhiều khối. Mẫu này phổ biến cho tất cả các mạng, từ VGG đến ResNeXt. Thực vậy, để thiết kế các mạng AnyNet chung, Radosavovic.Kosaraju.Girshick.ea.2020 đã sử dụng khối ResNeXt của [fig_resnext_block](#fig_resnext_block).


![Không gian thiết kế AnyNet. Các số $(\mathit{c}, \mathit{r})$ dọc theo mỗi mũi tên chỉ số kênh $c$ và độ phân giải $\mathit{r} \times \mathit{r}$ của ảnh tại điểm đó. Từ trái sang phải: cấu trúc mạng chung bao gồm thân, thân chính và đầu; thân chính gồm bốn giai đoạn; cấu trúc chi tiết của một giai đoạn; hai cấu trúc thay thế cho các khối, một không có giảm mẫu và một giảm một nửa độ phân giải theo mỗi chiều. Các lựa chọn thiết kế bao gồm độ sâu $\mathit{d_i}$, số kênh đầu ra $\mathit{c_i}$, số nhóm $\mathit{g_i}$, và tỷ lệ nút cổ chai $\mathit{k_i}$ cho bất kỳ giai đoạn nào $\mathit{i}$.](../img/anynet.svg)
<a id="fig_anynet_full"></a>

Hãy xem xét cấu trúc được phác thảo trong [fig_anynet_full](#fig_anynet_full) một cách chi tiết. Như đã đề cập, một AnyNet bao gồm thân, thân chính và đầu. Thân nhận đầu vào là ảnh RGB (3 kênh), sử dụng tích chập $3 \times 3$ với sải bước là $2$, theo sau là chuẩn hóa batch, để giảm một nửa độ phân giải từ $r \times r$ xuống $r/2 \times r/2$. Hơn nữa, nó tạo ra $c_0$ kênh đóng vai trò làm đầu vào cho thân chính.

Vì mạng được thiết kế để hoạt động tốt với ảnh ImageNet có hình dạng $224 \times 224 \times 3$, thân chính phục vụ để giảm điều này xuống $7 \times 7 \times c_4$ qua 4 giai đoạn (nhớ lại rằng $224 / 2^{1+4} = 7$), mỗi giai đoạn có sải bước cuối cùng là $2$. Cuối cùng, đầu sử dụng một thiết kế hoàn toàn tiêu chuẩn qua gộp trung bình toàn cục, tương tự như NiN ([sec_nin](#sec_nin)), theo sau là lớp kết nối đầy đủ để phát ra vector $n$ chiều cho phân loại $n$ lớp.

Hầu hết các quyết định thiết kế liên quan đều nằm trong thân chính của mạng. Nó tiến hành theo các giai đoạn, trong đó mỗi giai đoạn được tạo thành từ cùng loại khối ResNeXt như chúng ta đã thảo luận trong [subsec_resnext](#subsec_resnext). Thiết kế đó một lần nữa hoàn toàn chung: chúng ta bắt đầu với một khối giảm một nửa độ phân giải bằng cách sử dụng sải bước là $2$ (cái ngoài cùng bên phải trong [fig_anynet_full](#fig_anynet_full)). Để khớp với điều này, nhánh dư của khối ResNeXt cần đi qua một tích chập $1 \times 1$. Khối này được theo sau bởi một số biến đổi các khối ResNeXt bổ sung để lại cả độ phân giải và số kênh không thay đổi. Lưu ý rằng một thực hành thiết kế phổ biến là thêm một nút cổ chai nhỏ trong thiết kế các khối tích chập.
Như vậy, với tỷ lệ nút cổ chai $k_i \geq 1$ chúng ta có một số kênh, $c_i/k_i$, trong mỗi khối cho giai đoạn $i$ (như các thí nghiệm cho thấy, điều này không thực sự hiệu quả và nên bỏ qua). Cuối cùng, vì chúng ta đang xử lý các khối ResNeXt, chúng ta cũng cần chọn số nhóm $g_i$ cho tích chập theo nhóm tại giai đoạn $i$.

Không gian thiết kế có vẻ chung này tuy nhiên cung cấp cho chúng ta nhiều tham số: chúng ta có thể đặt chiều rộng khối (số kênh) $c_0, \ldots c_4$, độ sâu (số khối) mỗi giai đoạn $d_1, \ldots d_4$, tỷ lệ nút cổ chai $k_1, \ldots k_4$, và chiều rộng nhóm (số nhóm) $g_1, \ldots g_4$.
Tổng cộng lên đến 17 tham số, dẫn đến số lượng cấu hình không hợp lý lớn mà sẽ cần khám phá. Chúng ta cần một số công cụ để giảm hiệu quả không gian thiết kế khổng lồ này. Đây là nơi vẻ đẹp khái niệm của không gian thiết kế xuất hiện. Trước khi làm điều đó, hãy triển khai thiết kế chung trước.


```python
class AnyNet(d2l.Classifier):
    def stem(self, num_channels):
        return nn.Sequential(
            nn.LazyConv2d(num_channels, kernel_size=3, stride=2, padding=1),
            nn.LazyBatchNorm2d(), nn.ReLU())
```


Mỗi giai đoạn bao gồm `depth` khối ResNeXt,
trong đó `num_channels` chỉ định chiều rộng khối.
Lưu ý rằng khối đầu tiên giảm một nửa chiều cao và chiều rộng của ảnh đầu vào.


```python
@d2l.add_to_class(AnyNet)
def stage(self, depth, num_channels, groups, bot_mul):
    blk = []
    for i in range(depth):
        if i == 0:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul,
                use_1x1conv=True, strides=2))
        else:
            blk.append(d2l.ResNeXtBlock(num_channels, groups, bot_mul))
    return nn.Sequential(*blk)
```


Lắp ráp thân mạng, thân chính và đầu lại với nhau,
chúng ta hoàn thành triển khai AnyNet.


## Phân phối và Tham số của Không gian Thiết kế

Như vừa thảo luận trong [subsec_the-anynet-design-space](#subsec_the-anynet-design-space), các tham số của một không gian thiết kế là các siêu tham số của các mạng trong không gian thiết kế đó.
Hãy xem xét vấn đề xác định các tham số tốt trong không gian thiết kế AnyNet. Chúng ta có thể cố gắng tìm *lựa chọn tham số tốt nhất duy nhất* cho một lượng tính toán nhất định (ví dụ: FLOPs và thời gian tính toán). Nếu chúng ta cho phép chỉ *hai* lựa chọn có thể cho mỗi tham số, chúng ta sẽ phải khám phá $2^{17} = 131072$ tổ hợp để tìm giải pháp tốt nhất. Điều này rõ ràng là không khả thi do chi phí quá lớn. Còn tệ hơn, chúng ta thực sự không học được gì từ bài tập này về cách thiết kế mạng. Lần tiếp theo chúng ta thêm, chẳng hạn, một giai đoạn X, hoặc một phép toán dịch chuyển, hoặc tương tự, chúng ta sẽ phải bắt đầu lại từ đầu. Còn tệ hơn nữa, do tính ngẫu nhiên trong huấn luyện (làm tròn, xáo trộn, lỗi bit), không có hai lần chạy nào có khả năng tạo ra kết quả giống hệt nhau. Một chiến lược tốt hơn là cố gắng xác định các hướng dẫn chung về cách các lựa chọn tham số nên được liên kết với nhau. Ví dụ, tỷ lệ nút cổ chai, số kênh, khối, nhóm, hoặc sự thay đổi của chúng giữa các lớp nên lý tưởng là được điều chỉnh bởi một tập hợp các quy tắc đơn giản. Cách tiếp cận trong radosavovic2019network dựa trên bốn giả định sau:

1. Chúng ta giả định rằng các nguyên tắc thiết kế chung thực sự tồn tại, vì vậy nhiều mạng thỏa mãn các yêu cầu này nên cung cấp hiệu suất tốt. Do đó, xác định một *phân phối* trên các mạng có thể là một chiến lược hợp lý. Nói cách khác, chúng ta giả định rằng có nhiều cây kim tốt trong đống cỏ khô.
1. Chúng ta không cần huấn luyện các mạng đến hội tụ trước khi đánh giá liệu một mạng có tốt không. Thay vào đó, đủ để sử dụng kết quả trung gian như hướng dẫn đáng tin cậy cho độ chính xác cuối cùng. Sử dụng các proxy (gần đúng) để tối ưu hóa một mục tiêu được gọi là tối ưu hóa đa độ trung thực [forrester2007multi]. Do đó, tối ưu hóa thiết kế được thực hiện, dựa trên độ chính xác đạt được chỉ sau vài lần qua tập dữ liệu, giảm đáng kể chi phí.
1. Kết quả thu được ở quy mô nhỏ hơn (cho các mạng nhỏ hơn) tổng quát hóa cho các mạng lớn hơn. Do đó, tối ưu hóa được thực hiện cho các mạng có cấu trúc tương tự, nhưng với số lượng khối nhỏ hơn, ít kênh hơn, v.v. Chỉ ở cuối cùng chúng ta mới cần xác minh rằng các mạng tìm được cũng cung cấp hiệu suất tốt ở quy mô lớn.
1. Các khía cạnh của thiết kế có thể được xấp xỉ nhân tử hóa để có thể suy ra ảnh hưởng của chúng đến chất lượng kết quả tương đối độc lập. Nói cách khác, bài toán tối ưu hóa có độ khó vừa phải.

Những giả định này cho phép chúng ta kiểm tra nhiều mạng với chi phí thấp. Đặc biệt, chúng ta có thể *lấy mẫu* đều từ không gian cấu hình và đánh giá hiệu suất của chúng. Sau đó, chúng ta có thể đánh giá chất lượng của lựa chọn tham số bằng cách xem xét *phân phối* của sai số/độ chính xác có thể đạt được với các mạng nói trên. Ký hiệu $F(e)$ là hàm phân phối tích lũy (CDF) cho các lỗi được gây ra bởi các mạng của một không gian thiết kế nhất định, được rút ra bằng phân phối xác suất $p$. Tức là,

$$F(e, p) \stackrel{\textrm{def}}{=} P_{\textrm{net} \sim p} \{e(\textrm{net}) \leq e\}.$$

Mục tiêu của chúng ta bây giờ là tìm một phân phối $p$ trên *các mạng* sao cho hầu hết các mạng có tỷ lệ lỗi rất thấp và nơi hỗ trợ của $p$ là súc tích. Tất nhiên, điều này không khả thi về mặt tính toán để thực hiện chính xác. Chúng ta dùng một mẫu mạng $\mathcal{Z} \stackrel{\textrm{def}}{=} \{\textrm{net}_1, \ldots \textrm{net}_n\}$ (với các lỗi $e_1, \ldots, e_n$ tương ứng) từ $p$ và sử dụng CDF thực nghiệm $\hat{F}(e, \mathcal{Z})$ thay thế:

$$\hat{F}(e, \mathcal{Z}) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(e_i \leq e).$$

Bất cứ khi nào CDF cho một tập hợp lựa chọn chiếm ưu thế (hoặc khớp) với CDF khác thì theo đó lựa chọn tham số của nó là vượt trội (hoặc không có sự khác biệt). Theo đó
Radosavovic.Kosaraju.Girshick.ea.2020 đã thử nghiệm với tỷ lệ nút cổ chai mạng chia sẻ $k_i = k$ cho tất cả các giai đoạn $i$ của mạng. Điều này loại bỏ ba trong số bốn tham số điều chỉnh tỷ lệ nút cổ chai. Để đánh giá xem điều này có (tiêu cực) ảnh hưởng đến hiệu suất không, người ta có thể rút các mạng từ phân phối bị ràng buộc và không bị ràng buộc và so sánh các CDF tương ứng. Hóa ra ràng buộc này không ảnh hưởng đến độ chính xác của phân phối mạng chút nào, như có thể thấy trong bảng đầu tiên của [fig_regnet-fig](#fig_regnet-fig).
Tương tự, chúng ta có thể chọn lấy cùng chiều rộng nhóm $g_i = g$ xuất hiện tại các giai đoạn khác nhau của mạng. Một lần nữa, điều này không ảnh hưởng đến hiệu suất, như có thể thấy trong bảng thứ hai của [fig_regnet-fig](#fig_regnet-fig).
Cả hai bước kết hợp giảm số tham số tự do đi sáu.

![So sánh các hàm phân phối tích lũy lỗi thực nghiệm của không gian thiết kế. $\textrm{AnyNet}_\mathit{A}$ là không gian thiết kế gốc; $\textrm{AnyNet}_\mathit{B}$ buộc các tỷ lệ nút cổ chai, $\textrm{AnyNet}_\mathit{C}$ cũng buộc chiều rộng nhóm, $\textrm{AnyNet}_\mathit{D}$ tăng độ sâu mạng qua các giai đoạn. Từ trái sang phải: (i) buộc tỷ lệ nút cổ chai không ảnh hưởng đến hiệu suất; (ii) buộc chiều rộng nhóm không ảnh hưởng đến hiệu suất; (iii) tăng chiều rộng mạng (kênh) qua các giai đoạn cải thiện hiệu suất; (iv) tăng độ sâu mạng qua các giai đoạn cải thiện hiệu suất. Hình ảnh được cung cấp bởi Radosavovic.Kosaraju.Girshick.ea.2020.](../img/regnet-fig.png)
<a id="fig_regnet-fig"></a>

Tiếp theo chúng ta tìm cách giảm bớt sự đa dạng của các lựa chọn có thể cho chiều rộng và độ sâu của các giai đoạn. Đây là một giả định hợp lý rằng, khi chúng ta đi sâu hơn, số kênh nên tăng, tức là $c_i \geq c_{i-1}$ ($w_{i+1} \geq w_i$ theo ký hiệu của họ trong [fig_regnet-fig](#fig_regnet-fig)), tạo ra
$\textrm{AnyNetX}_D$. Tương tự, cũng hợp lý khi giả định rằng khi các giai đoạn tiến triển, chúng nên trở nên sâu hơn, tức là $d_i \geq d_{i-1}$, tạo ra $\textrm{AnyNetX}_E$. Điều này có thể được xác minh thực nghiệm trong bảng thứ ba và thứ tư của [fig_regnet-fig](#fig_regnet-fig) tương ứng.

## RegNet

Không gian thiết kế $\textrm{AnyNetX}_E$ kết quả bao gồm các mạng đơn giản
tuân theo các nguyên tắc thiết kế dễ hiểu:

* Chia sẻ tỷ lệ nút cổ chai $k_i = k$ cho tất cả các giai đoạn $i$;
* Chia sẻ chiều rộng nhóm $g_i = g$ cho tất cả các giai đoạn $i$;
* Tăng chiều rộng mạng qua các giai đoạn: $c_{i} \leq c_{i+1}$;
* Tăng độ sâu mạng qua các giai đoạn: $d_{i} \leq d_{i+1}$.

Điều này để lại cho chúng ta một tập hợp lựa chọn cuối cùng: cách chọn các giá trị cụ thể cho các tham số trên của không gian thiết kế $\textrm{AnyNetX}_E$ cuối cùng. Bằng cách nghiên cứu các mạng có hiệu suất tốt nhất từ phân phối trong $\textrm{AnyNetX}_E$, có thể quan sát thấy những điều sau: chiều rộng của mạng lý tưởng tăng tuyến tính với chỉ số khối trên mạng, tức là $c_j \approx c_0 + c_a j$, trong đó $j$ là chỉ số khối và độ dốc $c_a > 0$. Vì chúng ta chỉ được chọn chiều rộng khối khác nhau mỗi giai đoạn, chúng ta đến một hàm từng đoạn hằng, được thiết kế để khớp với sự phụ thuộc này. Hơn nữa, các thí nghiệm cũng cho thấy tỷ lệ nút cổ chai $k = 1$ hoạt động tốt nhất, tức là chúng ta được khuyên không sử dụng nút cổ chai chút nào.

Chúng tôi khuyến nghị độc giả quan tâm xem xét thêm chi tiết trong thiết kế các mạng cụ thể cho các lượng tính toán khác nhau bằng cách đọc Radosavovic.Kosaraju.Girshick.ea.2020. Ví dụ, một biến thể RegNetX 32 lớp hiệu quả được cho bởi $k = 1$ (không có nút cổ chai), $g = 16$ (chiều rộng nhóm là 16), $c_1 = 32$ và $c_2 = 80$ kênh cho giai đoạn đầu và thứ hai tương ứng, được chọn là $d_1=4$ và $d_2=6$ khối sâu. Nhận thức đáng ngạc nhiên từ thiết kế là nó vẫn áp dụng, ngay cả khi điều tra các mạng ở quy mô lớn hơn. Còn tốt hơn, nó thậm chí giữ đúng cho các thiết kế mạng Squeeze-and-Excitation (SE) (RegNetY) có activation kênh toàn cục [Hu.Shen.Sun.2018].


Chúng ta có thể thấy rằng mỗi giai đoạn RegNetX dần dần giảm độ phân giải và tăng các kênh đầu ra.


## Huấn luyện

Huấn luyện RegNetX 32 lớp trên tập dữ liệu Fashion-MNIST cũng giống như trước đây.


## Thảo luận

Với các thiên kiến quy nạp mong muốn (giả định hoặc ưu tiên) như tính cục bộ và bất biến dịch chuyển ([sec_why-conv](#sec_why-conv))
cho thị giác, CNN đã là các kiến trúc thống trị trong lĩnh vực này. Điều này vẫn đúng từ LeNet cho đến khi Transformer ([sec_transformer](#sec_transformer)) [Dosovitskiy.Beyer.Kolesnikov.ea.2021, touvron2021training] bắt đầu vượt qua CNN về độ chính xác. Mặc dù phần lớn tiến bộ gần đây về Transformer thị giác *có thể* được chuyển ngược lại vào CNN [liu2022convnet], nhưng chỉ có thể làm được với chi phí tính toán cao hơn. Cũng quan trọng không kém, các tối ưu hóa phần cứng gần đây (NVIDIA Ampere và Hopper) chỉ mở rộng khoảng cách có lợi cho Transformer.

Đáng chú ý là Transformer có mức độ thiên kiến quy nạp thấp hơn đáng kể đối với tính cục bộ và bất biến dịch chuyển so với CNN. Việc các cấu trúc được học chiến thắng là do, không kém phần quan trọng, sự sẵn có của các bộ sưu tập ảnh lớn, chẳng hạn như LAION-400m và LAION-5B [schuhmann2022laion] với tới 5 tỷ ảnh. Khá ngạc nhiên, một số công việc liên quan hơn trong bối cảnh này thậm chí bao gồm MLP [tolstikhin2021mlp].

Tóm lại, Transformer thị giác ([sec_vision-transformer](#sec_vision-transformer)) hiện dẫn đầu về
hiệu suất tiên tiến trong phân loại ảnh quy mô lớn,
cho thấy rằng *tính mở rộng đánh bại thiên kiến quy nạp* [Dosovitskiy.Beyer.Kolesnikov.ea.2021].
Điều này bao gồm tiền huấn luyện các Transformer quy mô lớn ([sec_large-pretraining-transformers](#sec_large-pretraining-transformers)) với tự attention đa đầu ([sec_multihead-attention](#sec_multihead-attention)). Chúng tôi mời độc giả đi sâu vào các chương này để thảo luận chi tiết hơn nhiều.

## Bài tập

1. Tăng số giai đoạn lên bốn. Bạn có thể thiết kế một RegNetX sâu hơn hoạt động tốt hơn không?
1. Bỏ ResNeXt khỏi RegNet bằng cách thay thế khối ResNeXt bằng khối ResNet. Mô hình mới của bạn hoạt động như thế nào?
1. Triển khai nhiều thể hiện của họ "VioNet" bằng cách *vi phạm* các nguyên tắc thiết kế của RegNetX. Chúng hoạt động như thế nào? Cái nào trong ($d_i$, $c_i$, $g_i$, $b_i$) là yếu tố quan trọng nhất?
1. Mục tiêu của bạn là thiết kế "MLP hoàn hảo". Bạn có thể sử dụng các nguyên tắc thiết kế được giới thiệu ở trên để tìm kiến trúc tốt không? Có thể ngoại suy từ các mạng nhỏ sang các mạng lớn không?


[Thảo luận](https://discuss.d2l.ai/t/7463)
