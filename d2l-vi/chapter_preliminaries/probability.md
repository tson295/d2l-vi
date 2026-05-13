# Xác suất và Thống kê
<a id="sec_prob"></a>

Dù theo cách nào,
machine learning đều xoay quanh sự không chắc chắn.
Trong học có giám sát, ta muốn dự đoán
một điều gì đó chưa biết (*mục tiêu*)
dựa trên điều gì đó đã biết (*đặc trưng*).
Tùy thuộc vào mục tiêu của ta,
ta có thể cố gắng dự đoán
giá trị có khả năng nhất của mục tiêu.
Hoặc ta có thể dự đoán giá trị có
khoảng cách kỳ vọng nhỏ nhất từ mục tiêu.
Và đôi khi ta không chỉ muốn
dự đoán một giá trị cụ thể
mà còn muốn *lượng hóa sự không chắc chắn của ta*.
Ví dụ, dựa trên một số đặc trưng
mô tả bệnh nhân,
ta có thể muốn biết *khả năng* họ
bị đau tim trong năm tới là bao nhiêu.
Trong học không giám sát,
ta thường quan tâm đến sự không chắc chắn.
Để xác định xem một tập hợp các phép đo có bất thường không,
sẽ hữu ích khi biết khả năng
quan sát các giá trị trong quần thể quan tâm.
Hơn nữa, trong học tăng cường,
ta muốn phát triển các tác tử
hành động thông minh trong các môi trường khác nhau.
Điều này đòi hỏi suy luận về
cách môi trường có thể thay đổi
và phần thưởng nào ta có thể nhận được
tương ứng với mỗi hành động có thể.

*Xác suất* là lĩnh vực toán học
liên quan đến suy luận trong điều kiện không chắc chắn.
Với mô hình xác suất của một quá trình nào đó,
ta có thể suy luận về khả năng xảy ra của các sự kiện khác nhau.
Việc dùng xác suất để mô tả
tần suất của các sự kiện có thể lặp lại
(như tung đồng xu)
khá không gây tranh cãi.
Trên thực tế, các học giả *tần suất* tuân theo
cách hiểu về xác suất
chỉ áp dụng cho các sự kiện có thể lặp lại như vậy.
Ngược lại, các học giả *Bayesian*
dùng ngôn ngữ xác suất rộng hơn
để hình thức hóa suy luận trong điều kiện không chắc chắn.
Xác suất Bayesian được đặc trưng
bởi hai tính năng độc đáo:
(i) gán mức độ tin tưởng
cho các sự kiện không thể lặp lại,
ví dụ: *xác suất* một con đập sẽ sụp đổ là bao nhiêu?;
và (ii) tính chủ quan. Trong khi xác suất Bayesian
cung cấp các quy tắc rõ ràng
về cách cập nhật niềm tin
dưới ánh sáng của bằng chứng mới,
nó cho phép các cá nhân khác nhau
bắt đầu với các niềm tin *tiên nghiệm* khác nhau.
*Thống kê* giúp ta suy luận ngược lại,
bắt đầu từ việc thu thập và tổ chức dữ liệu
và rút ra những suy luận
ta có thể đưa ra về quá trình
đã tạo ra dữ liệu.
Bất cứ khi nào ta phân tích tập dữ liệu, tìm kiếm các mẫu
mà ta hy vọng có thể đặc trưng cho quần thể rộng hơn,
ta đang sử dụng tư duy thống kê.
Nhiều khóa học, chuyên ngành, luận văn, nghề nghiệp, bộ môn,
công ty và tổ chức đã được dành riêng
cho việc nghiên cứu xác suất và thống kê.
Mặc dù phần này chỉ chạm bề mặt,
ta sẽ cung cấp nền tảng
mà bạn cần để bắt đầu xây dựng mô hình.


```python
%matplotlib inline
from d2l import torch as d2l
import random
import torch
from torch.distributions.multinomial import Multinomial
```


## Ví dụ Đơn giản: Tung Đồng xu

Hãy tưởng tượng ta định tung một đồng xu
và muốn lượng hóa khả năng
ta thấy mặt ngửa (so với mặt úp).
Nếu đồng xu *cân bằng*,
thì cả hai kết quả
(ngửa và úp),
đều có xác suất bằng nhau.
Hơn nữa, nếu ta định tung đồng xu $n$ lần
thì tỷ lệ mặt ngửa
mà ta *kỳ vọng* sẽ thấy
phải khớp chính xác với
tỷ lệ *kỳ vọng* của mặt úp.
Một cách trực quan để thấy điều này
là qua tính đối xứng:
với mỗi kết quả có thể
với $n_\textrm{h}$ lần ngửa và $n_\textrm{t} = (n - n_\textrm{h})$ lần úp,
có một kết quả có xác suất bằng nhau
với $n_\textrm{t}$ lần ngửa và $n_\textrm{h}$ lần úp.
Lưu ý rằng điều này chỉ có thể
nếu trung bình ta kỳ vọng thấy
$1/2$ lần tung ra ngửa
và $1/2$ ra úp.
Tất nhiên, nếu bạn thực hiện thí nghiệm này
nhiều lần với $n=1000000$ lần tung mỗi lần,
bạn có thể không bao giờ thấy một thử nghiệm
mà $n_\textrm{h} = n_\textrm{t}$ chính xác.

Chính thức, đại lượng $1/2$ được gọi là *xác suất*
và ở đây nó nắm bắt sự chắc chắn mà với đó
bất kỳ lần tung nào sẽ ra mặt ngửa.
Xác suất gán điểm từ $0$ đến $1$
cho các kết quả quan tâm, được gọi là *sự kiện*.
Ở đây sự kiện quan tâm là $\textrm{ngửa}$
và ta ký hiệu xác suất tương ứng là $P(\textrm{ngửa})$.
Xác suất $1$ chỉ sự chắc chắn tuyệt đối
(hãy tưởng tượng đồng xu hai mặt đều là ngửa)
và xác suất $0$ chỉ sự không thể xảy ra
(ví dụ: nếu cả hai mặt đều là úp).
Các tần suất $n_\textrm{h}/n$ và $n_\textrm{t}/n$ không phải là xác suất
mà là *thống kê*.
Xác suất là các đại lượng *lý thuyết*
nằm bên dưới quá trình tạo dữ liệu.
Ở đây, xác suất $1/2$
là thuộc tính của chính đồng xu.
Ngược lại, thống kê là các đại lượng *thực nghiệm*
được tính là hàm của dữ liệu quan sát.
Mối quan tâm của ta đối với các đại lượng xác suất và thống kê
gắn bó không thể tách rời.
Ta thường thiết kế các thống kê đặc biệt gọi là *ước lượng*
mà, dựa trên tập dữ liệu, tạo ra các *ước lượng*
về các tham số mô hình như xác suất.
Hơn nữa, khi những ước lượng đó thỏa mãn
một thuộc tính tốt gọi là *nhất quán*,
các ước lượng của ta sẽ hội tụ
về xác suất tương ứng.
Đổi lại, những xác suất suy ra này
nói về các thuộc tính thống kê có thể có
của dữ liệu từ cùng quần thể
mà ta có thể gặp trong tương lai.

Giả sử ta tình cờ tìm thấy một đồng xu thực
mà ta không biết
giá trị thực của $P(\textrm{ngửa})$.
Để điều tra đại lượng này
bằng các phương pháp thống kê,
ta cần (i) thu thập một số dữ liệu;
và (ii) thiết kế một ước lượng.
Thu thập dữ liệu ở đây rất dễ;
ta có thể tung đồng xu nhiều lần
và ghi lại tất cả kết quả.
Chính thức, việc lấy các thực hiện
từ một quá trình ngẫu nhiên cơ bản nào đó
được gọi là *lấy mẫu*.
Như bạn có thể đã đoán,
một ước lượng tự nhiên
là tỷ lệ của
số lần quan sát *ngửa*
trên tổng số lần tung.

Bây giờ, giả sử đồng xu thực sự cân bằng,
tức là $P(\textrm{ngửa}) = 0.5$.
Để mô phỏng các lần tung đồng xu cân bằng,
ta có thể gọi bất kỳ bộ tạo số ngẫu nhiên nào.
Có một số cách dễ để lấy mẫu
sự kiện với xác suất $0.5$.
Ví dụ `random.random` của Python
tạo ra các số trong khoảng $[0,1]$
trong đó xác suất nằm
trong bất kỳ khoảng con $[a, b] \subset [0,1]$ nào
bằng $b-a$.
Do đó ta có thể lấy ra `0` và `1` với xác suất `0.5` mỗi loại
bằng cách kiểm tra xem số thực được trả về có lớn hơn `0.5` không:

```python
num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])
```

Tổng quát hơn, ta có thể mô phỏng nhiều lần rút
từ bất kỳ biến nào với số hữu hạn
các kết quả có thể
(như lần tung đồng xu hoặc con xúc xắc)
bằng cách gọi hàm multinomial,
đặt đối số đầu tiên
là số lần rút
và đối số thứ hai là danh sách xác suất
liên quan đến mỗi kết quả có thể.
Để mô phỏng mười lần tung đồng xu cân bằng,
ta gán vector xác suất `[0.5, 0.5]`,
diễn giải chỉ số 0 là ngửa
và chỉ số 1 là úp.
Hàm trả về vector
có độ dài bằng số
kết quả có thể (ở đây là 2),
trong đó thành phần đầu tiên cho ta biết
số lần xuất hiện ngửa
và thành phần thứ hai cho ta biết
số lần xuất hiện úp.


```python
fair_probs = torch.tensor([0.5, 0.5])
Multinomial(100, fair_probs).sample()
```


Mỗi lần bạn chạy quá trình lấy mẫu này,
bạn sẽ nhận được một giá trị ngẫu nhiên mới
có thể khác với kết quả trước.
Chia cho số lần tung
cho ta *tần suất*
của mỗi kết quả trong dữ liệu.
Lưu ý rằng các tần suất này,
giống như các xác suất
mà chúng định ước lượng, có tổng bằng $1$.


```python
Multinomial(100, fair_probs).sample() / 100
```


Ở đây, mặc dù đồng xu mô phỏng của ta cân bằng
(ta tự đặt xác suất `[0.5, 0.5]`),
số lần ngửa và úp có thể không giống hệt nhau.
Đó là vì ta chỉ rút một số mẫu tương đối nhỏ.
Nếu ta không tự triển khai mô phỏng,
mà chỉ thấy kết quả,
làm sao ta biết đồng xu có hơi lệch không
hay nếu độ lệch có thể từ $1/2$
chỉ là hiệu ứng của cỡ mẫu nhỏ?
Hãy xem điều gì xảy ra khi ta mô phỏng 10.000 lần tung.


```python
counts = Multinomial(10000, fair_probs).sample()
counts / 10000
```


Nói chung, đối với trung bình của các sự kiện lặp lại (như tung đồng xu),
khi số lần lặp lại tăng,
các ước lượng của ta được đảm bảo hội tụ
về xác suất cơ bản thực sự.
Công thức toán học của hiện tượng này
được gọi là *luật số lớn*
và *định lý giới hạn trung tâm*
cho ta biết rằng trong nhiều tình huống,
khi cỡ mẫu $n$ tăng,
các lỗi này nên giảm
ở tốc độ $(1/\sqrt{n})$.
Hãy có thêm trực giác bằng cách nghiên cứu
cách ước lượng của ta phát triển khi ta tăng
số lần tung từ 1 đến 10.000.

```python
counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()

d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```


Mỗi đường cong liền nét tương ứng với một trong hai giá trị của đồng xu
và cho ta xác suất ước lượng rằng đồng xu ra giá trị đó
sau mỗi nhóm thí nghiệm.
Đường đứt đoạn màu đen cho biết xác suất cơ bản thực sự.
Khi ta có thêm dữ liệu bằng cách thực hiện nhiều thí nghiệm hơn,
các đường cong hội tụ về xác suất thực.
Bạn có thể đã bắt đầu thấy hình dạng
của một số câu hỏi nâng cao hơn
mà các nhà thống kê bận tâm:
Quá trình hội tụ này xảy ra nhanh như thế nào?
Nếu ta đã kiểm tra nhiều đồng xu
được sản xuất tại cùng một nhà máy,
ta có thể kết hợp thông tin này như thế nào?

## Xử lý Chính thức Hơn

Ta đã đi khá xa: đặt ra
mô hình xác suất,
tạo dữ liệu tổng hợp,
chạy ước lượng thống kê,
đánh giá thực nghiệm sự hội tụ,
và báo cáo số liệu lỗi (kiểm tra độ lệch).
Tuy nhiên, để đi xa hơn nhiều,
ta cần chính xác hơn.

Khi xử lý ngẫu nhiên,
ta ký hiệu tập hợp các kết quả có thể là $\mathcal{S}$
và gọi nó là *không gian mẫu* hoặc *không gian kết quả*.
Ở đây, mỗi phần tử là một *kết quả* có thể phân biệt.
Trong trường hợp tung một đồng xu,
$\mathcal{S} = \{\textrm{ngửa}, \textrm{úp}\}$.
Với một con xúc xắc, $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$.
Khi tung hai đồng xu, các kết quả có thể là
$\{(\textrm{ngửa}, \textrm{ngửa}), (\textrm{ngửa}, \textrm{úp}), (\textrm{úp}, \textrm{ngửa}),  (\textrm{úp}, \textrm{úp})\}$.
*Sự kiện* là tập con của không gian mẫu.
Ví dụ, sự kiện "lần tung đầu tiên ra ngửa"
tương ứng với tập hợp $\{(\textrm{ngửa}, \textrm{ngửa}), (\textrm{ngửa}, \textrm{úp})\}$.
Bất cứ khi nào kết quả $z$ của thí nghiệm ngẫu nhiên thỏa mãn
$z \in \mathcal{A}$, thì sự kiện $\mathcal{A}$ đã xảy ra.
Với một lần tung xúc xắc, ta có thể định nghĩa các sự kiện
"thấy số $5$" ($\mathcal{A} = \{5\}$)
và "thấy số lẻ" ($\mathcal{B} = \{1, 3, 5\}$).
Trong trường hợp này, nếu xúc xắc ra số $5$,
ta nói rằng cả $\mathcal{A}$ và $\mathcal{B}$ đều xảy ra.
Mặt khác, nếu $z = 3$,
thì $\mathcal{A}$ không xảy ra
nhưng $\mathcal{B}$ xảy ra.

*Hàm xác suất* ánh xạ các sự kiện
sang các giá trị thực ${P: \mathcal{A} \subseteq \mathcal{S} \rightarrow [0,1]}$.
Xác suất, ký hiệu $P(\mathcal{A})$, của sự kiện $\mathcal{A}$
trong không gian mẫu $\mathcal{S}$ đã cho,
có các tính chất sau:

* Xác suất của bất kỳ sự kiện $\mathcal{A}$ nào là số thực không âm, tức là $P(\mathcal{A}) \geq 0$;
* Xác suất của toàn bộ không gian mẫu là $1$, tức là $P(\mathcal{S}) = 1$;
* Với bất kỳ dãy đếm được các sự kiện $\mathcal{A}_1, \mathcal{A}_2, \ldots$ *loại trừ lẫn nhau* (tức là $\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ với mọi $i \neq j$), xác suất để bất kỳ sự kiện nào trong số chúng xảy ra bằng tổng xác suất riêng lẻ của chúng, tức là $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$.

Các tiên đề của lý thuyết xác suất này,
được đề xuất bởi Kolmogorov.1933,
có thể được áp dụng để nhanh chóng rút ra một số hệ quả quan trọng.
Ví dụ, từ đó ngay lập tức suy ra
rằng xác suất để sự kiện $\mathcal{A}$ bất kỳ
*hoặc* phần bù $\mathcal{A}'$ của nó xảy ra là 1
(vì $\mathcal{A} \cup \mathcal{A}' = \mathcal{S}$).
Ta cũng có thể chứng minh rằng $P(\emptyset) = 0$
vì $1 = P(\mathcal{S} \cup \mathcal{S}') = P(\mathcal{S} \cup \emptyset) = P(\mathcal{S}) + P(\emptyset) = 1 + P(\emptyset)$.
Do đó, xác suất để sự kiện $\mathcal{A}$ bất kỳ
*và* phần bù $\mathcal{A}'$ của nó xảy ra đồng thời
là $P(\mathcal{A} \cap \mathcal{A}') = 0$.
Nói không chính thức, điều này cho ta biết rằng các sự kiện không thể
có xác suất xảy ra bằng không.


## Biến Ngẫu nhiên

Khi ta nói về các sự kiện như tung xúc xắc
ra số lẻ hay lần tung đồng xu đầu tiên ra ngửa,
ta đang gợi lên ý tưởng về một *biến ngẫu nhiên*.
Chính thức, biến ngẫu nhiên là các ánh xạ
từ không gian mẫu cơ bản
sang một tập hợp (có thể là nhiều) giá trị.
Bạn có thể tự hỏi biến ngẫu nhiên
khác gì với không gian mẫu,
vì cả hai đều là tập hợp kết quả.
Quan trọng là, biến ngẫu nhiên có thể thô hơn nhiều
so với không gian mẫu thô.
Ta có thể định nghĩa biến ngẫu nhiên nhị phân như "lớn hơn 0.5"
ngay cả khi không gian mẫu cơ bản là vô hạn,
ví dụ: các điểm trên đoạn thẳng giữa $0$ và $1$.
Ngoài ra, nhiều biến ngẫu nhiên
có thể chia sẻ cùng không gian mẫu cơ bản.
Ví dụ "liệu báo động nhà tôi có kêu không"
và "liệu nhà tôi có bị trộm không"
đều là các biến ngẫu nhiên nhị phân
chia sẻ không gian mẫu cơ bản.
Do đó, biết giá trị của một biến ngẫu nhiên
có thể cho ta biết điều gì đó về giá trị có thể của biến ngẫu nhiên khác.
Biết rằng báo động đã kêu,
ta có thể nghi ngờ rằng nhà có thể đã bị trộm.

Mỗi giá trị được lấy bởi biến ngẫu nhiên tương ứng
với một tập con của không gian mẫu cơ bản.
Do đó, sự xuất hiện mà biến ngẫu nhiên $X$
nhận giá trị $v$, ký hiệu là $X=v$, là một *sự kiện*
và $P(X=v)$ biểu thị xác suất của nó.
Đôi khi ký hiệu này có thể cồng kềnh,
và ta có thể lạm dụng ký hiệu khi ngữ cảnh rõ ràng.
Ví dụ, ta có thể dùng $P(X)$ để chỉ rộng rãi
*phân phối* của $X$, tức là
hàm cho ta biết xác suất
$X$ nhận bất kỳ giá trị nào.
Các lúc khác ta viết các biểu thức
như $P(X,Y) = P(X) P(Y)$,
như cách viết tắt để biểu thị một phát biểu
đúng với tất cả các giá trị
mà các biến ngẫu nhiên $X$ và $Y$ có thể nhận, tức là
với mọi $i,j$ ta có $P(X=i \textrm{ và } Y=j) = P(X=i)P(Y=j)$.
Các lúc khác, ta lạm dụng ký hiệu bằng cách viết
$P(v)$ khi biến ngẫu nhiên rõ ràng từ ngữ cảnh.
Vì một sự kiện trong lý thuyết xác suất là tập hợp kết quả từ không gian mẫu,
ta có thể chỉ định một phạm vi giá trị để biến ngẫu nhiên nhận.
Ví dụ, $P(1 \leq X \leq 3)$ biểu thị xác suất của sự kiện $\{1 \leq X \leq 3\}$.

Lưu ý rằng có sự khác biệt tinh tế
giữa các biến ngẫu nhiên *rời rạc*,
như lần tung đồng xu hoặc tung xúc xắc,
và các biến *liên tục*,
như cân nặng và chiều cao của một người
được lấy mẫu ngẫu nhiên từ quần thể.
Trong trường hợp này ta hiếm khi thực sự quan tâm đến
chiều cao chính xác của ai đó.
Hơn nữa, nếu ta thực hiện các phép đo đủ chính xác,
ta sẽ thấy rằng không có hai người nào trên hành tinh
có chiều cao giống hệt nhau.
Trên thực tế, với các phép đo đủ tinh,
bạn sẽ không bao giờ có cùng chiều cao
khi thức dậy và khi đi ngủ.
Hỏi về xác suất chính xác của ai đó
cao 1.801392782910287192 mét không có nhiều ý nghĩa.
Thay vào đó, ta thường quan tâm hơn đến khả năng nói
liệu chiều cao của ai đó có nằm trong khoảng đã cho không,
chẳng hạn giữa 1.79 và 1.81 mét.
Trong những trường hợp này ta làm việc với *mật độ* xác suất.
Chiều cao chính xác 1.80 mét
không có xác suất, nhưng mật độ khác không.
Để tính xác suất được gán cho một khoảng,
ta phải lấy *tích phân* của mật độ
trên khoảng đó.

## Nhiều Biến Ngẫu nhiên

Bạn có thể nhận thấy rằng ta thậm chí không thể
đi qua phần trước mà không
đưa ra các phát biểu liên quan đến tương tác
giữa nhiều biến ngẫu nhiên
(nhớ lại rằng $P(X,Y) = P(X) P(Y)$).
Hầu hết machine learning
liên quan đến các mối quan hệ như vậy.
Ở đây, không gian mẫu sẽ là
quần thể quan tâm,
ví dụ như khách hàng giao dịch với doanh nghiệp,
ảnh trên Internet,
hoặc protein được các nhà sinh học biết đến.
Mỗi biến ngẫu nhiên sẽ biểu diễn
giá trị (chưa biết) của một thuộc tính khác nhau.
Bất cứ khi nào ta lấy mẫu một cá thể từ quần thể,
ta quan sát một thực hiện của mỗi biến ngẫu nhiên.
Vì các giá trị được lấy bởi biến ngẫu nhiên
tương ứng với các tập con của không gian mẫu
có thể chồng lên nhau, chồng lên nhau một phần,
hoặc hoàn toàn rời nhau,
biết giá trị của một biến ngẫu nhiên
có thể khiến ta cập nhật niềm tin của mình
về giá trị nào của biến ngẫu nhiên khác là có thể.
Nếu bệnh nhân vào bệnh viện
và ta quan sát thấy họ
đang khó thở
và mất khứu giác,
thì ta tin rằng họ có nhiều khả năng
mắc COVID-19 hơn ta có thể
nếu họ không khó thở
và khứu giác hoàn toàn bình thường.

Khi làm việc với nhiều biến ngẫu nhiên,
ta có thể xây dựng các sự kiện tương ứng
với mọi tổ hợp giá trị
mà các biến có thể nhận cùng nhau.
Hàm xác suất gán
xác suất cho mỗi tổ hợp này
(ví dụ: $A=a$ và $B=b$)
được gọi là hàm *xác suất kết hợp*
và đơn giản trả về xác suất được gán
cho giao của các tập con tương ứng
của không gian mẫu.
*Xác suất kết hợp* được gán cho sự kiện
mà các biến ngẫu nhiên $A$ và $B$
nhận giá trị $a$ và $b$ tương ứng,
được ký hiệu là $P(A = a, B = b)$,
trong đó dấu phẩy chỉ "và".
Lưu ý rằng với giá trị $a$ và $b$ bất kỳ,
từ đó suy ra

$$P(A=a, B=b) \leq P(A=a) \textrm{ và } P(A=a, B=b) \leq P(B = b),$$

vì để $A=a$ và $B=b$ xảy ra,
$A=a$ phải xảy ra *và* $B=b$ cũng phải xảy ra.
Thú vị là, xác suất kết hợp
cho ta biết tất cả những gì ta có thể biết về các
biến ngẫu nhiên này theo nghĩa xác suất,
và có thể được dùng để suy ra nhiều đại lượng hữu ích khác,
bao gồm khôi phục các
phân phối riêng lẻ $P(A)$ và $P(B)$.
Để khôi phục $P(A=a)$ ta chỉ cần tính tổng
$P(A=a, B=v)$ theo tất cả các giá trị $v$
mà biến ngẫu nhiên $B$ có thể nhận:
$P(A=a) = \sum_v P(A=a, B=v)$.

Tỷ lệ $\frac{P(A=a, B=b)}{P(A=a)} \leq 1$
hóa ra cực kỳ quan trọng.
Nó được gọi là *xác suất có điều kiện*,
và được ký hiệu qua ký tự "$\mid$":

$$P(B=b \mid A=a) = P(A=a,B=b)/P(A=a).$$

Nó cho ta biết xác suất mới
liên quan đến sự kiện $B=b$,
sau khi ta điều kiện hóa trên thực tế $A=a$ đã xảy ra.
Ta có thể coi xác suất có điều kiện này
là hạn chế sự chú ý chỉ vào tập con
của không gian mẫu liên quan đến $A=a$
và sau đó chuẩn hóa lại để
tất cả xác suất tổng bằng 1.
Xác suất có điều kiện
thực ra chỉ là các xác suất thông thường
và do đó tuân thủ tất cả các tiên đề,
miễn là ta điều kiện hóa tất cả các số hạng
theo cùng sự kiện và do đó
hạn chế sự chú ý vào cùng không gian mẫu.
Ví dụ, với các sự kiện rời nhau
$\mathcal{B}$ và $\mathcal{B}'$, ta có
$P(\mathcal{B} \cup \mathcal{B}' \mid A = a) = P(\mathcal{B} \mid A = a) + P(\mathcal{B}' \mid A = a)$.

Sử dụng định nghĩa về xác suất có điều kiện,
ta có thể rút ra kết quả nổi tiếng gọi là *định lý Bayes*.
Theo định nghĩa, ta có $P(A, B) = P(B\mid A) P(A)$
và $P(A, B) = P(A\mid B) P(B)$.
Kết hợp cả hai phương trình cho ta
$P(B\mid A) P(A) = P(A\mid B) P(B)$ và do đó

$$P(A \mid B) = \frac{P(B\mid A) P(A)}{P(B)}.$$

Phương trình đơn giản này có những hệ quả sâu sắc vì
nó cho phép ta đảo ngược thứ tự điều kiện hóa.
Nếu ta biết cách ước lượng $P(B\mid A)$, $P(A)$ và $P(B)$,
thì ta có thể ước lượng $P(A\mid B)$.
Ta thường thấy dễ dàng ước lượng một số hạng trực tiếp
nhưng không phải số hạng kia và định lý Bayes có thể đến giải cứu ở đây.
Ví dụ, nếu ta biết tỷ lệ phổ biến của các triệu chứng đối với một bệnh nhất định,
và tỷ lệ phổ biến tổng thể của bệnh và triệu chứng tương ứng,
ta có thể xác định khả năng ai đó
mắc bệnh dựa trên các triệu chứng của họ.
Trong một số trường hợp ta có thể không có quyền truy cập trực tiếp vào $P(B)$,
chẳng hạn như tỷ lệ phổ biến của triệu chứng.
Trong trường hợp này một phiên bản đơn giản hóa của định lý Bayes có ích:

$$P(A \mid B) \propto P(B \mid A) P(A).$$

Vì ta biết rằng $P(A \mid B)$ phải được chuẩn hóa về $1$, tức là $\sum_a P(A=a \mid B) = 1$,
ta có thể dùng nó để tính

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{\sum_a P(B \mid A=a) P(A = a)}.$$

Trong thống kê Bayesian, ta coi người quan sát
là sở hữu một số niềm tin tiên nghiệm (chủ quan)
về khả năng của các giả thuyết có sẵn
được mã hóa trong *tiên nghiệm* $P(H)$,
và *hàm khả năng* cho biết khả năng
quan sát bất kỳ giá trị nào của bằng chứng thu thập được
cho mỗi giả thuyết trong lớp $P(E \mid H)$.
Định lý Bayes sau đó được giải thích là cho ta biết
cách cập nhật *tiên nghiệm* ban đầu $P(H)$
dưới ánh sáng của bằng chứng có sẵn $E$
để tạo ra niềm tin *hậu nghiệm*
$P(H \mid E) = \frac{P(E \mid H) P(H)}{P(E)}$.
Nói không chính thức, điều này có thể được phát biểu là
"hậu nghiệm bằng tiên nghiệm nhân khả năng, chia cho bằng chứng".
Bây giờ, vì bằng chứng $P(E)$ giống nhau với tất cả giả thuyết,
ta có thể thoát ra bằng cách đơn giản là chuẩn hóa trên các giả thuyết.

Lưu ý rằng $\sum_a P(A=a \mid B) = 1$ cũng cho phép ta *lề hóa* theo các biến ngẫu nhiên. Tức là, ta có thể bỏ các biến khỏi phân phối kết hợp như $P(A, B)$. Xét cho cùng, ta có

$$\sum_a P(B \mid A=a) P(A=a) = \sum_a P(B, A=a) = P(B).$$

Độc lập là một khái niệm quan trọng cơ bản khác
tạo thành xương sống của
nhiều ý tưởng quan trọng trong thống kê.
Nói ngắn gọn, hai biến là *độc lập*
nếu điều kiện hóa theo giá trị của $A$ không
gây ra bất kỳ thay đổi nào trong phân phối xác suất
liên quan đến $B$ và ngược lại.
Chính thức hơn, sự độc lập, ký hiệu là $A \perp B$,
đòi hỏi rằng $P(A \mid B) = P(A)$ và do đó,
rằng $P(A,B) = P(A \mid B) P(B) = P(A) P(B)$.
Sự độc lập thường là một giả định phù hợp.
Ví dụ, nếu biến ngẫu nhiên $A$
biểu diễn kết quả từ lần tung một đồng xu cân bằng
và biến ngẫu nhiên $B$
biểu diễn kết quả từ lần tung một đồng xu khác,
thì biết liệu $A$ ra ngửa
không nên ảnh hưởng đến xác suất
$B$ ra ngửa.

Sự độc lập đặc biệt hữu ích khi nó giữ giữa các lần rút liên tiếp
dữ liệu của ta từ phân phối cơ bản nào đó
(cho phép ta đưa ra kết luận thống kê mạnh)
hoặc khi nó giữ giữa các biến khác nhau trong dữ liệu của ta,
cho phép ta làm việc với các mô hình đơn giản hơn
mã hóa cấu trúc độc lập này.
Mặt khác, ước lượng các phụ thuộc
giữa các biến ngẫu nhiên thường là mục tiêu chính của học.
Ta quan tâm đến ước lượng xác suất của bệnh dựa trên triệu chứng
chính xác vì ta tin
rằng bệnh và triệu chứng *không* độc lập.

Lưu ý rằng vì xác suất có điều kiện là các xác suất thông thường,
các khái niệm về độc lập và phụ thuộc cũng áp dụng cho chúng.
Hai biến ngẫu nhiên $A$ và $B$ là *độc lập có điều kiện*
cho một biến thứ ba $C$ khi và chỉ khi $P(A, B \mid C) = P(A \mid C)P(B \mid C)$.
Thú vị là, hai biến có thể độc lập nói chung
nhưng trở nên phụ thuộc khi điều kiện hóa trên biến thứ ba.
Điều này thường xảy ra khi hai biến ngẫu nhiên $A$ và $B$
tương ứng với các nguyên nhân của biến thứ ba $C$.
Ví dụ, xương gãy và ung thư phổi có thể độc lập
trong quần thể nói chung nhưng nếu ta điều kiện hóa là đang ở bệnh viện
thì ta có thể thấy rằng xương gãy có tương quan âm với ung thư phổi.
Đó là vì xương gãy *giải thích tại sao* một người đang ở bệnh viện
và do đó giảm xác suất rằng họ nhập viện vì ung thư phổi.

Và ngược lại, hai biến ngẫu nhiên phụ thuộc
có thể trở nên độc lập khi điều kiện hóa trên biến thứ ba.
Điều này thường xảy ra khi hai sự kiện không liên quan
có một nguyên nhân chung.
Cỡ giày và trình độ đọc có tương quan cao
trong học sinh tiểu học,
nhưng tương quan này biến mất nếu ta điều kiện hóa theo tuổi.


## Một Ví dụ
<a id="subsec_probability_hiv_app"></a>

Hãy kiểm tra các kỹ năng của ta.
Giả sử một bác sĩ thực hiện xét nghiệm HIV cho bệnh nhân.
Xét nghiệm này khá chính xác và chỉ thất bại với xác suất 1%
nếu bệnh nhân khỏe mạnh nhưng được báo cáo là bệnh,
tức là bệnh nhân khỏe mạnh xét nghiệm dương tính trong 1% trường hợp.
Hơn nữa, nó không bao giờ không phát hiện HIV nếu bệnh nhân thực sự có HIV.
Ta dùng $D_1 \in \{0, 1\}$ để chỉ ra chẩn đoán
($0$ nếu âm tính và $1$ nếu dương tính)
và $H \in \{0, 1\}$ để biểu thị tình trạng HIV.

| Xác suất có điều kiện | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_1 = 1 \mid H)$        |     1 |  0.01 |
| $P(D_1 = 0 \mid H)$        |     0 |  0.99 |

Lưu ý rằng các tổng cột đều là 1 (nhưng tổng hàng không phải),
vì chúng là xác suất có điều kiện.
Hãy tính xác suất bệnh nhân có HIV
nếu xét nghiệm trả về dương tính, tức là $P(H = 1 \mid D_1 = 1)$.
Theo trực giác, điều này sẽ phụ thuộc vào tỷ lệ phổ biến của bệnh,
vì nó ảnh hưởng đến số lượng cảnh báo sai.
Giả sử quần thể khá không có bệnh, ví dụ: $P(H=1) = 0.0015$.
Để áp dụng định lý Bayes, ta cần áp dụng lề hóa
để xác định

$$\begin{aligned}
P(D_1 = 1)
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

Điều này dẫn ta đến

$$P(H = 1 \mid D_1 = 1) = \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} = 0.1306.$$

Nói cách khác, chỉ có 13.06% cơ hội
bệnh nhân thực sự có HIV,
mặc dù xét nghiệm khá chính xác.
Như ta thấy, xác suất có thể phản trực giác.
Bệnh nhân nên làm gì khi nhận được tin tức đáng sợ như vậy?
Có lẽ, bệnh nhân sẽ yêu cầu bác sĩ
thực hiện thêm một xét nghiệm khác để làm rõ.
Xét nghiệm thứ hai có các đặc điểm khác nhau
và không tốt bằng xét nghiệm đầu tiên.

| Xác suất có điều kiện | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_2 = 1 \mid H)$          |  0.98 |  0.03 |
| $P(D_2 = 0 \mid H)$          |  0.02 |  0.97 |

Thật không may, xét nghiệm thứ hai cũng trả về dương tính.
Hãy tính các xác suất cần thiết để áp dụng định lý Bayes
bằng cách giả định độc lập có điều kiện:

$$\begin{aligned}
P(D_1 = 1, D_2 = 1 \mid H = 0)
& = P(D_1 = 1 \mid H = 0) P(D_2 = 1 \mid H = 0)
=& 0.0003, \\
P(D_1 = 1, D_2 = 1 \mid H = 1)
& = P(D_1 = 1 \mid H = 1) P(D_2 = 1 \mid H = 1)
=& 0.98.
\end{aligned}
$$

Bây giờ ta có thể áp dụng lề hóa để thu được xác suất
cả hai xét nghiệm đều trả về dương tính:

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1)\\
&= P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
&= P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
&= 0.00176955.
\end{aligned}
$$

Cuối cùng, xác suất bệnh nhân có HIV khi cả hai xét nghiệm đều dương tính là

$$P(H = 1 \mid D_1 = 1, D_2 = 1)
= \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)}
= 0.8307.$$

Tức là, xét nghiệm thứ hai giúp ta đạt được độ tin cậy cao hơn nhiều rằng không phải mọi thứ đều ổn.
Mặc dù xét nghiệm thứ hai kém chính xác hơn đáng kể so với xét nghiệm đầu tiên,
nó vẫn cải thiện đáng kể ước lượng của ta.
Giả định cả hai xét nghiệm độc lập có điều kiện với nhau
là rất quan trọng cho khả năng tạo ra ước lượng chính xác hơn của ta.
Hãy lấy trường hợp cực đoan khi ta chạy cùng một xét nghiệm hai lần.
Trong tình huống này ta sẽ kỳ vọng cùng kết quả cả hai lần,
do đó không có thêm thông tin nào từ việc chạy lại cùng xét nghiệm.
Người đọc sắc sảo có thể nhận thấy rằng chẩn đoán hoạt động
như một bộ phân loại ẩn mình ngay trước mắt
trong đó khả năng của ta để quyết định liệu bệnh nhân có khỏe mạnh không
tăng lên khi ta thu được nhiều đặc trưng hơn (kết quả xét nghiệm).


## Kỳ vọng

Thường thì, việc ra quyết định đòi hỏi không chỉ nhìn vào
các xác suất được gán cho các sự kiện riêng lẻ
mà còn kết hợp chúng thành các tổng hợp hữu ích
có thể cung cấp cho ta sự hướng dẫn.
Ví dụ, khi biến ngẫu nhiên nhận giá trị vô hướng liên tục,
ta thường muốn biết giá trị nào để kỳ vọng *trung bình*.
Đại lượng này được gọi chính thức là *kỳ vọng*.
Nếu ta đang thực hiện đầu tư,
đại lượng quan tâm đầu tiên
có thể là lợi nhuận ta có thể kỳ vọng,
lấy trung bình qua tất cả kết quả có thể
(và tính trọng số theo xác suất thích hợp).
Ví dụ, giả sử với xác suất 50%,
một khoản đầu tư có thể thất bại hoàn toàn,
với xác suất 40% nó có thể cung cấp lợi nhuận $2\times$,
và với xác suất 10% nó có thể cung cấp lợi nhuận $10\times$.
Để tính lợi nhuận kỳ vọng,
ta tổng qua tất cả các lợi nhuận, nhân mỗi
với xác suất chúng sẽ xảy ra.
Điều này tạo ra kỳ vọng
$0.5 \cdot 0 + 0.4 \cdot 2 + 0.1 \cdot 10 = 1.8$.
Do đó lợi nhuận kỳ vọng là $1.8\times$.

Nói chung, *kỳ vọng* (hoặc trung bình)
của biến ngẫu nhiên $X$ được định nghĩa là

$$E[X] = E_{x \sim P}[x] = \sum_{x} x P(X = x).$$

Tương tự, với mật độ ta thu được $E[X] = \int x \;dp(x)$.
Đôi khi ta quan tâm đến giá trị kỳ vọng
của một hàm nào đó của $x$.
Ta có thể tính các kỳ vọng này như sau

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x) \textrm{ và } E_{x \sim P}[f(x)] = \int f(x) p(x) \;dx$$

cho xác suất rời rạc và mật độ tương ứng.
Quay lại ví dụ đầu tư ở trên,
$f$ có thể là *tiện ích* (hạnh phúc)
liên quan đến lợi nhuận.
Các nhà kinh tế học hành vi từ lâu đã nhận thấy
rằng con người liên quan đến sự bất tiện lớn hơn
với việc mất tiền so với tiện ích thu được
từ việc kiếm một đô la so với mức cơ sở của họ.
Hơn nữa, giá trị của tiền có xu hướng dưới tuyến tính.
Có 100 nghìn đô la so với không có đồng nào
có thể tạo ra sự khác biệt giữa trả tiền thuê,
ăn uống tốt và hưởng chăm sóc y tế chất lượng
so với chịu đựng cảnh vô gia cư.
Mặt khác, lợi ích từ việc có
200 nghìn so với 100 nghìn ít ấn tượng hơn.
Suy luận như vậy thúc đẩy câu nói sáo rỗng
rằng "tiện ích của tiền là logarit".

Nếu tiện ích liên quan đến mất toàn bộ là $-1$,
và các tiện ích liên quan đến lợi nhuận $1$, $2$ và $10$
lần lượt là $1$, $2$ và $4$,
thì hạnh phúc kỳ vọng của đầu tư
sẽ là $0.5 \cdot (-1) + 0.4 \cdot 2 + 0.1 \cdot 4 = 0.7$
(tổn thất tiện ích kỳ vọng 30%).
Nếu đây thực sự là hàm tiện ích của bạn,
bạn có thể tốt nhất là để tiền trong ngân hàng.

Đối với các quyết định tài chính,
ta cũng có thể muốn đo
mức độ *rủi ro* của khoản đầu tư.
Ở đây, ta không chỉ quan tâm đến giá trị kỳ vọng
mà còn đến mức độ biến động của các giá trị thực tế
so với giá trị này.
Lưu ý rằng ta không thể chỉ lấy
kỳ vọng của hiệu
giữa giá trị thực và giá trị kỳ vọng.
Đó là vì kỳ vọng của hiệu
là hiệu của kỳ vọng,
tức là $E[X - E[X]] = E[X] - E[E[X]] = 0$.
Tuy nhiên, ta có thể xét kỳ vọng
của bất kỳ hàm không âm nào của hiệu này.
*Phương sai* của biến ngẫu nhiên được tính bằng cách xét
giá trị kỳ vọng của bình phương hiệu:

$$\textrm{Var}[X] = E\left[(X - E[X])^2\right] = E[X^2] - E[X]^2.$$

Ở đây đẳng thức được suy từ việc khai triển
$(X - E[X])^2 = X^2 - 2 X E[X] + E[X]^2$
và lấy kỳ vọng cho từng số hạng.
Căn bậc hai của phương sai là một đại lượng hữu ích khác
gọi là *độ lệch chuẩn*.
Mặc dù nó và phương sai
truyền đạt cùng một thông tin (cái này có thể tính từ cái kia),
độ lệch chuẩn có thuộc tính tốt
là được biểu diễn cùng đơn vị
với đại lượng gốc được biểu diễn
bởi biến ngẫu nhiên.

Cuối cùng, phương sai của một hàm
của biến ngẫu nhiên
được định nghĩa tương tự là

$$\textrm{Var}_{x \sim P}[f(x)] = E_{x \sim P}[f^2(x)] - E_{x \sim P}[f(x)]^2.$$

Quay lại ví dụ đầu tư của ta,
bây giờ ta có thể tính phương sai của khoản đầu tư.
Nó được cho bởi $0.5 \cdot 0 + 0.4 \cdot 2^2 + 0.1 \cdot 10^2 - 1.8^2 = 8.36$.
Về mọi mục đích thực tế đây là một khoản đầu tư rủi ro.
Lưu ý rằng theo quy ước toán học, trung bình và phương sai
thường được tham chiếu là $\mu$ và $\sigma^2$.
Điều này đặc biệt đúng khi ta dùng chúng
để tham số hóa phân phối Gaussian.

Giống như ta đã giới thiệu kỳ vọng
và phương sai cho các biến ngẫu nhiên *vô hướng*,
ta có thể làm như vậy cho các biến có giá trị vector.
Kỳ vọng dễ dàng, vì ta có thể áp dụng chúng theo từng phần tử.
Ví dụ, $\boldsymbol{\mu} \stackrel{\textrm{def}}{=} E_{\mathbf{x} \sim P}[\mathbf{x}]$
có tọa độ $\mu_i = E_{\mathbf{x} \sim P}[x_i]$.
*Hiệp phương sai* phức tạp hơn.
Ta định nghĩa chúng bằng cách lấy kỳ vọng của *tích ngoài*
của hiệu giữa biến ngẫu nhiên và trung bình của chúng:

$$\boldsymbol{\Sigma} \stackrel{\textrm{def}}{=} \textrm{Cov}_{\mathbf{x} \sim P}[\mathbf{x}] = E_{\mathbf{x} \sim P}\left[(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top\right].$$

Ma trận $\boldsymbol{\Sigma}$ này được gọi là ma trận hiệp phương sai.
Một cách dễ dàng để thấy tác dụng của nó là xét một vector $\mathbf{v}$ nào đó
có cùng kích thước với $\mathbf{x}$.
Từ đó suy ra

$$\mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v} = E_{\mathbf{x} \sim P}\left[\mathbf{v}^\top(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{v}\right] = \textrm{Var}_{x \sim P}[\mathbf{v}^\top \mathbf{x}].$$

Như vậy, $\boldsymbol{\Sigma}$ cho phép ta tính phương sai
cho bất kỳ hàm tuyến tính nào của $\mathbf{x}$
bằng phép nhân ma trận đơn giản.
Các phần tử ngoài đường chéo cho ta biết các tọa độ tương quan như thế nào:
giá trị 0 có nghĩa là không có tương quan,
trong khi giá trị dương lớn hơn
có nghĩa là chúng tương quan mạnh hơn.


## Thảo luận

Trong machine learning, có nhiều điều không chắc chắn!
Ta có thể không chắc chắn về giá trị của nhãn khi cho đặc trưng.
Ta có thể không chắc chắn về giá trị ước lượng của tham số.
Ta thậm chí có thể không chắc chắn liệu dữ liệu đến trong quá trình triển khai
có từ cùng phân phối như dữ liệu huấn luyện không.

Bởi *bất định aleatoric*, ta có nghĩa là sự không chắc chắn
vốn có trong bài toán,
và do tính ngẫu nhiên thực sự
không được tính bởi các biến quan sát.
Bởi *bất định epistemic*, ta có nghĩa là sự không chắc chắn
về các tham số của mô hình, loại không chắc chắn
mà ta có thể hy vọng giảm bằng cách thu thập thêm dữ liệu.
Ta có thể có bất định epistemic
về xác suất
đồng xu ra ngửa,
nhưng ngay cả khi ta biết xác suất này,
ta vẫn còn bất định aleatoric
về kết quả của bất kỳ lần tung nào trong tương lai.
Dù ta xem ai đó tung đồng xu cân bằng bao lâu,
ta sẽ không bao giờ chắc chắn hơn hoặc ít hơn 50%
rằng lần tung tiếp theo sẽ ra ngửa.
Các thuật ngữ này xuất phát từ mô hình hóa cơ học,
(xem ví dụ Der-Kiureghian.Ditlevsen.2009 để có bài đánh giá về khía cạnh này của [lượng hóa bất định](https://en.wikipedia.org/wiki/Uncertainty_quantification)).
Đáng chú ý là, tuy nhiên, rằng các thuật ngữ này cấu thành một sự lạm dụng nhẹ ngôn ngữ.
Thuật ngữ *epistemic* đề cập đến bất cứ điều gì liên quan đến *kiến thức*
và do đó, theo nghĩa triết học, tất cả sự không chắc chắn đều là epistemic.

Ta đã thấy rằng lấy mẫu dữ liệu từ phân phối xác suất chưa biết nào đó
có thể cung cấp cho ta thông tin có thể được dùng để ước lượng
các tham số của phân phối tạo dữ liệu.
Điều đó nói, tốc độ mà điều này có thể xảy ra có thể khá chậm.
Trong ví dụ tung đồng xu của ta (và nhiều ví dụ khác)
ta không thể làm tốt hơn là thiết kế các ước lượng
hội tụ ở tốc độ $1/\sqrt{n}$,
trong đó $n$ là cỡ mẫu (ví dụ: số lần tung).
Điều này có nghĩa là bằng cách đi từ 10 đến 1000 quan sát (thường là nhiệm vụ rất có thể đạt được)
ta thấy giảm mười lần sự không chắc chắn,
trong khi 1000 quan sát tiếp theo ít giúp đỡ hơn theo tỷ lệ,
chỉ giảm 1.41 lần.
Đây là tính năng bền vững của machine learning:
trong khi thường có những lợi ích dễ dàng, cần một lượng dữ liệu rất lớn,
và thường kèm theo đó là một lượng tính toán khổng lồ, để đạt được thêm lợi ích.
Để có đánh giá thực nghiệm về thực tế này cho các mô hình ngôn ngữ quy mô lớn, xem Revels.Lubin.Papamarkou.2016.

Ta cũng đã làm sắc nét ngôn ngữ và công cụ của mình cho mô hình hóa thống kê.
Trong quá trình đó, ta đã học về xác suất có điều kiện
và về một trong những phương trình quan trọng nhất trong thống kê---định lý Bayes.
Đây là công cụ hiệu quả để tách rời thông tin được truyền đạt bởi dữ liệu
qua số hạng khả năng $P(B \mid A)$ giải quyết
các quan sát $B$ khớp với lựa chọn tham số $A$ tốt như thế nào,
và xác suất tiên nghiệm $P(A)$ điều chỉnh mức độ hợp lý
một lựa chọn cụ thể của $A$ như thế nào ngay từ đầu.
Đặc biệt, ta đã thấy cách quy tắc này có thể được áp dụng
để gán xác suất cho các chẩn đoán,
dựa trên hiệu quả của xét nghiệm *và*
tỷ lệ phổ biến của chính bệnh đó (tức là tiên nghiệm của ta).

Cuối cùng, ta đã giới thiệu tập câu hỏi không tầm thường đầu tiên
về tác dụng của một phân phối xác suất cụ thể,
cụ thể là kỳ vọng và phương sai.
Mặc dù có nhiều hơn chỉ kỳ vọng tuyến tính và bậc hai
cho một phân phối xác suất,
hai đại lượng này đã cung cấp khá nhiều kiến thức
về hành vi có thể của phân phối.
Ví dụ, [bất đẳng thức Chebyshev](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality)
phát biểu rằng $P(|X - \mu| \geq k \sigma) \leq 1/k^2$,
trong đó $\mu$ là kỳ vọng, $\sigma^2$ là phương sai của phân phối,
và $k > 1$ là tham số tin cậy do ta chọn.
Nó cho ta biết rằng các lần rút từ phân phối nằm
với xác suất ít nhất 50%
trong khoảng $[-\sqrt{2} \sigma, \sqrt{2} \sigma]$
căn giữa theo kỳ vọng.


## Bài tập

1. Cho ví dụ về quan sát thêm dữ liệu có thể giảm lượng không chắc chắn về kết quả xuống mức tùy ý thấp.
1. Cho ví dụ về việc quan sát thêm dữ liệu chỉ giảm lượng không chắc chắn đến một điểm nhất định và sau đó không giảm thêm. Giải thích tại sao điều này xảy ra và bạn kỳ vọng điểm này ở đâu.
1. Ta đã minh họa thực nghiệm sự hội tụ về giá trị trung bình đối với lần tung đồng xu. Hãy tính phương sai của ước lượng xác suất ta thấy mặt ngửa sau khi rút $n$ mẫu.
    1. Phương sai biến đổi như thế nào theo số quan sát?
    1. Dùng bất đẳng thức Chebyshev để giới hạn độ lệch so với kỳ vọng.
    1. Nó liên quan đến định lý giới hạn trung tâm như thế nào?
1. Giả sử ta rút $m$ mẫu $x_i$ từ phân phối xác suất với trung bình bằng không và phương sai đơn vị. Tính trung bình $z_m \stackrel{\textrm{def}}{=} m^{-1} \sum_{i=1}^m x_i$. Ta có thể áp dụng bất đẳng thức Chebyshev cho từng $z_m$ một cách độc lập không? Tại sao không?
1. Cho hai sự kiện với xác suất $P(\mathcal{A})$ và $P(\mathcal{B})$, hãy tính cận trên và cận dưới của $P(\mathcal{A} \cup \mathcal{B})$ và $P(\mathcal{A} \cap \mathcal{B})$. Gợi ý: hình dung tình huống bằng [biểu đồ Venn](https://en.wikipedia.org/wiki/Venn_diagram).
1. Giả sử ta có dãy biến ngẫu nhiên, chẳng hạn $A$, $B$ và $C$, trong đó $B$ chỉ phụ thuộc vào $A$, và $C$ chỉ phụ thuộc vào $B$, bạn có thể đơn giản hóa xác suất kết hợp $P(A, B, C)$ không? Gợi ý: đây là [chuỗi Markov](https://en.wikipedia.org/wiki/Markov_chain).
1. Trong [subsec_probability_hiv_app](#subsec_probability_hiv_app), giả sử rằng các kết quả của hai xét nghiệm không độc lập. Đặc biệt giả sử rằng mỗi xét nghiệm riêng lẻ có tỷ lệ dương tính giả 10% và tỷ lệ âm tính giả 1%. Tức là, giả sử $P(D =1 \mid H=0) = 0.1$ và $P(D = 0 \mid H=1) = 0.01$. Hơn nữa, giả sử rằng với $H = 1$ (nhiễm) các kết quả xét nghiệm độc lập có điều kiện, tức là $P(D_1, D_2 \mid H=1) = P(D_1 \mid H=1) P(D_2 \mid H=1)$ nhưng với bệnh nhân khỏe mạnh các kết quả được liên kết qua $P(D_1 = D_2 = 1 \mid H=0) = 0.02$.
    1. Hãy xây dựng bảng xác suất kết hợp cho $D_1$ và $D_2$, với $H=0$ dựa trên thông tin bạn có cho đến nay.
    1. Tính xác suất bệnh nhân bị bệnh ($H=1$) sau khi một xét nghiệm trả về dương tính. Bạn có thể giả sử xác suất cơ sở giống như trước $P(H=1) = 0.0015$.
    1. Tính xác suất bệnh nhân bị bệnh ($H=1$) sau khi cả hai xét nghiệm trả về dương tính.
1. Giả sử bạn là quản lý tài sản cho ngân hàng đầu tư và bạn có lựa chọn cổ phiếu $s_i$ để đầu tư. Danh mục đầu tư của bạn cần có tổng bằng $1$ với trọng số $\alpha_i$ cho mỗi cổ phiếu. Các cổ phiếu có lợi nhuận trung bình $\boldsymbol{\mu} = E_{\mathbf{s} \sim P}[\mathbf{s}]$ và hiệp phương sai $\boldsymbol{\Sigma} = \textrm{Cov}_{\mathbf{s} \sim P}[\mathbf{s}]$.
    1. Tính lợi nhuận kỳ vọng cho danh mục $\boldsymbol{\alpha}$ đã cho.
    1. Nếu bạn muốn tối đa hóa lợi nhuận của danh mục, bạn nên chọn đầu tư như thế nào?
    1. Tính *phương sai* của danh mục.
    1. Hãy phát biểu bài toán tối ưu hóa tối đa hóa lợi nhuận trong khi giữ phương sai bị ràng buộc ở cận trên. Đây là [danh mục Markovitz](https://en.wikipedia.org/wiki/Markowitz_model) đoạt giải Nobel [Mangram.2013]. Để giải quyết nó, bạn sẽ cần bộ giải quy hoạch bậc hai, điều gì đó nằm ngoài phạm vi của cuốn sách này.


[Thảo luận](https://discuss.d2l.ai/t/37)
