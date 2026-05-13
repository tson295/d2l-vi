# Từ Lớp Kết nối Đầy đủ đến Tích chập
<a id="sec_why-conv"></a>

Cho đến nay,
các mô hình mà chúng ta đã thảo luận
vẫn là những lựa chọn phù hợp
khi chúng ta làm việc với dữ liệu dạng bảng.
Dạng bảng có nghĩa là dữ liệu bao gồm
các hàng tương ứng với các mẫu
và các cột tương ứng với các đặc trưng.
Với dữ liệu dạng bảng, chúng ta có thể dự đoán
rằng các mẫu chúng ta tìm kiếm có thể liên quan đến
các tương tác giữa các đặc trưng,
nhưng chúng ta không giả định bất kỳ cấu trúc *tiên nghiệm* nào
về cách các đặc trưng tương tác.

Đôi khi, chúng ta thực sự thiếu kiến thức để có thể hướng dẫn việc xây dựng các kiến trúc phức tạp hơn.
Trong những trường hợp này, một MLP
có thể là điều tốt nhất chúng ta có thể làm.
Tuy nhiên, đối với dữ liệu tri giác chiều cao,
các mạng không có cấu trúc như vậy có thể trở nên cồng kềnh.

Ví dụ, hãy quay lại ví dụ quen thuộc của chúng ta
về phân biệt mèo và chó.
Giả sử chúng ta làm tốt công việc thu thập dữ liệu,
thu thập một bộ dữ liệu có chú thích gồm các bức ảnh một megapixel.
Điều này có nghĩa là mỗi đầu vào của mạng có một triệu chiều.
Ngay cả khi giảm mạnh xuống còn một nghìn chiều ẩn
cũng cần một lớp kết nối đầy đủ
được đặc trưng bởi $10^6 \times 10^3 = 10^9$ tham số.
Trừ khi chúng ta có nhiều GPU, tài năng
cho tối ưu hóa phân tán,
và một lượng kiên nhẫn phi thường,
việc học các tham số của mạng này
có thể trở nên không khả thi.

Một người đọc cẩn thận có thể phản đối lập luận này
với cơ sở rằng độ phân giải một megapixel có thể không cần thiết.
Tuy nhiên, trong khi chúng ta có thể thoát khỏi
với một trăm nghìn pixel,
lớp ẩn kích thước 1000 của chúng ta đánh giá thấp nghiêm trọng
số lượng đơn vị ẩn cần thiết
để học các biểu diễn tốt của hình ảnh,
vì vậy một hệ thống thực tế vẫn sẽ cần hàng tỷ tham số.
Hơn nữa, việc học một bộ phân loại bằng cách khớp nhiều tham số như vậy
có thể yêu cầu thu thập một bộ dữ liệu khổng lồ.
Tuy nhiên ngày nay cả con người và máy tính đều có thể
phân biệt mèo và chó khá tốt,
có vẻ mâu thuẫn với những trực giác này.
Đó là vì hình ảnh thể hiện cấu trúc phong phú
có thể được khai thác bởi con người
và các mô hình machine learning như nhau.
Mạng nơ-ron tích chập (CNN) là một cách sáng tạo
mà machine learning đã áp dụng để khai thác
một số cấu trúc đã biết trong các hình ảnh tự nhiên.


## Bất biến

Hãy tưởng tượng rằng chúng ta muốn phát hiện một vật thể trong hình ảnh.
Có vẻ hợp lý rằng bất kỳ phương pháp nào
chúng ta sử dụng để nhận dạng vật thể không nên quá lo lắng
về vị trí chính xác của vật thể trong hình ảnh.
Lý tưởng nhất, hệ thống của chúng ta nên khai thác kiến thức này.
Lợn thường không bay và máy bay thường không bơi.
Tuy nhiên, chúng ta vẫn nên nhận ra
một con lợn dù nó xuất hiện ở đầu hình ảnh.
Chúng ta có thể lấy cảm hứng ở đây
từ trò chơi trẻ em "Waldo ở đâu"
(mà bản thân nó đã truyền cảm hứng cho nhiều bản sao thực tế, như được mô tả trong [img_waldo](#img_waldo)).
Trò chơi bao gồm một số cảnh hỗn loạn
đầy ắp các hoạt động.
Waldo xuất hiện ở đâu đó trong mỗi cảnh,
thường ẩn náu ở một vị trí khó ngờ.
Mục tiêu của người đọc là tìm anh ta.
Mặc dù trang phục đặc trưng của anh ta,
điều này có thể khó đến mức ngạc nhiên,
do số lượng lớn các yếu tố gây xao nhãng.
Tuy nhiên, *Waldo trông như thế nào*
không phụ thuộc vào *Waldo đang ở đâu*.
Chúng ta có thể quét hình ảnh bằng một bộ phát hiện Waldo
có thể gán điểm cho mỗi vùng nhỏ,
cho biết khả năng vùng đó chứa Waldo.
Trên thực tế, nhiều thuật toán phát hiện và phân đoạn đối tượng
dựa trên cách tiếp cận này [Long.Shelhamer.Darrell.2015].
CNN hệ thống hóa ý tưởng về *bất biến không gian* này,
khai thác nó để học các biểu diễn hữu ích
với ít tham số hơn.

![Bạn có thể tìm thấy Waldo (hình ảnh cảm ơn William Murphy (Infomatique))?](../img/waldo-football.jpg)
<a id="img_waldo"></a>

Bây giờ chúng ta có thể làm cho những trực giác này cụ thể hơn
bằng cách liệt kê một vài điều mong muốn để hướng dẫn thiết kế của chúng ta
về một kiến trúc mạng nơ-ron phù hợp cho thị giác máy tính:

1. Trong các lớp sớm nhất, mạng của chúng ta
   nên phản ứng tương tự với cùng một vùng nhỏ,
   bất kể nó xuất hiện ở đâu trong hình ảnh. Nguyên tắc này được gọi là *bất biến dịch chuyển* (hoặc *đồng biến dịch chuyển*).
1. Các lớp sớm nhất của mạng nên tập trung vào các vùng cục bộ,
   mà không quan tâm đến nội dung của hình ảnh ở các vùng xa. Đây là nguyên tắc *cục bộ*.
   Cuối cùng, các biểu diễn cục bộ này có thể được tổng hợp
   để đưa ra dự đoán ở cấp độ toàn bộ hình ảnh.
1. Khi tiến vào sâu hơn, các lớp sâu hơn nên có khả năng nắm bắt các đặc trưng tầm xa hơn của
   hình ảnh, theo cách tương tự như thị giác cấp cao hơn trong tự nhiên.

Hãy xem điều này được chuyển thành toán học như thế nào.


## Ràng buộc MLP

Để bắt đầu, chúng ta có thể xét một MLP
với hình ảnh hai chiều $\mathbf{X}$ làm đầu vào
và các biểu diễn ẩn ngay lập tức của chúng
$\mathbf{H}$ cũng được biểu diễn dưới dạng ma trận (chúng là các tensor hai chiều trong code),
trong đó cả $\mathbf{X}$ và $\mathbf{H}$ có cùng hình dạng.
Hãy để điều đó thấm vào.
Bây giờ chúng ta tưởng tượng rằng không chỉ đầu vào mà
cả các biểu diễn ẩn cũng có cấu trúc không gian.

Gọi $[\mathbf{X}]_{i, j}$ và $[\mathbf{H}]_{i, j}$ là điểm ảnh
tại vị trí $(i,j)$
trong ảnh đầu vào và biểu diễn ẩn, tương ứng.
Do đó, để mỗi đơn vị ẩn
nhận đầu vào từ mỗi điểm ảnh đầu vào,
chúng ta sẽ chuyển từ sử dụng ma trận trọng số
(như trước đây trong MLP)
sang biểu diễn các tham số của chúng ta
như tensor trọng số bậc bốn $\mathsf{W}$.
Giả sử $\mathbf{U}$ chứa hệ số chặn,
chúng ta có thể biểu diễn chính thức lớp kết nối đầy đủ như

$$\begin{aligned} \left[\mathbf{H}\right]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.\end{aligned}$$

Việc chuyển từ $\mathsf{W}$ sang $\mathsf{V}$ hoàn toàn chỉ là bề ngoài
vì có sự tương ứng một-một
giữa các hệ số trong cả hai tensor bậc bốn.
Chúng ta chỉ đơn giản lập chỉ số lại các chỉ số $(k, l)$
sao cho $k = i+a$ và $l = j+b$.
Nói cách khác, chúng ta đặt $[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$.
Các chỉ số $a$ và $b$ chạy qua cả độ lệch dương và âm,
bao phủ toàn bộ hình ảnh.
Đối với bất kỳ vị trí ($i$, $j$) nào trong biểu diễn ẩn $[\mathbf{H}]_{i, j}$,
chúng ta tính giá trị của nó bằng cách tổng hợp trên các điểm ảnh trong $x$,
lấy tâm xung quanh $(i, j)$ và được trọng số bởi $[\mathsf{V}]_{i, j, a, b}$. Trước khi tiếp tục, hãy xem xét tổng số tham số cần thiết cho một lớp *duy nhất* trong tham số hóa này: một hình ảnh $1000 \times 1000$ (1 megapixel) được ánh xạ tới một biểu diễn ẩn $1000 \times 1000$. Điều này yêu cầu $10^{12}$ tham số, vượt xa những gì máy tính hiện tại có thể xử lý.

### Bất biến Dịch chuyển

Bây giờ hãy áp dụng nguyên tắc đầu tiên
đã được thiết lập ở trên: bất biến dịch chuyển [Zhang.ea.1988].
Điều này có nghĩa là một sự dịch chuyển trong đầu vào $\mathbf{X}$
chỉ đơn giản dẫn đến một sự dịch chuyển trong biểu diễn ẩn $\mathbf{H}$.
Điều này chỉ có thể xảy ra nếu $\mathsf{V}$ và $\mathbf{U}$ thực sự không phụ thuộc vào $(i, j)$. Như vậy,
chúng ta có $[\mathsf{V}]_{i, j, a, b} = [\mathbf{V}]_{a, b}$ và $\mathbf{U}$ là một hằng số, giả sử $u$.
Kết quả là, chúng ta có thể đơn giản hóa định nghĩa cho $\mathbf{H}$:

$$[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$


Đây là một *phép tích chập*!
Chúng ta đang hiệu quả trọng số các điểm ảnh tại $(i+a, j+b)$
trong vùng lân cận của vị trí $(i, j)$ với các hệ số $[\mathbf{V}]_{a, b}$
để thu được giá trị $[\mathbf{H}]_{i, j}$.
Lưu ý rằng $[\mathbf{V}]_{a, b}$ cần ít hệ số hơn nhiều so với $[\mathsf{V}]_{i, j, a, b}$ vì nó
không còn phụ thuộc vào vị trí trong hình ảnh. Do đó, số lượng tham số cần thiết không còn là $10^{12}$ nữa mà là $4 \times 10^6$ hợp lý hơn nhiều: chúng ta vẫn còn sự phụ thuộc vào $a, b \in (-1000, 1000)$. Tóm lại, chúng ta đã đạt được tiến bộ đáng kể. Mạng nơ-ron trễ thời gian (TDNN) là một trong những ví dụ đầu tiên khai thác ý tưởng này [Waibel.Hanazawa.Hinton.ea.1989].

### Cục bộ

Bây giờ hãy áp dụng nguyên tắc thứ hai: cục bộ.
Như đã được thúc đẩy ở trên, chúng ta tin rằng không cần phải
nhìn rất xa vị trí $(i, j)$
để thu thập thông tin liên quan
để đánh giá những gì đang xảy ra tại $[\mathbf{H}]_{i, j}$.
Điều này có nghĩa là ngoài một phạm vi $|a|> \Delta$ hoặc $|b| > \Delta$,
chúng ta nên đặt $[\mathbf{V}]_{a, b} = 0$.
Tương đương, chúng ta có thể viết lại $[\mathbf{H}]_{i, j}$ là

$$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$

Điều này giảm số lượng tham số từ $4 \times 10^6$ xuống $4 \Delta^2$, trong đó $\Delta$ thường nhỏ hơn $10$. Như vậy, chúng ta đã giảm số lượng tham số thêm bốn bậc độ lớn. Lưu ý rằng :eqref:`eq_conv-layer`, là những gì được gọi, tóm lại, là một *lớp tích chập*.
*Mạng nơ-ron tích chập* (CNN)
là một họ đặc biệt của mạng nơ-ron chứa các lớp tích chập.
Trong cộng đồng nghiên cứu deep learning,
$\mathbf{V}$ được gọi là *nhân tích chập*,
một *bộ lọc*, hoặc đơn giản là *trọng số* của lớp là các tham số có thể học được.

Trong khi trước đây, chúng ta có thể cần hàng tỷ tham số
để biểu diễn chỉ một lớp trong một mạng xử lý hình ảnh,
bây giờ chúng ta thường chỉ cần vài trăm, mà không
thay đổi chiều của cả
đầu vào hay biểu diễn ẩn.
Cái giá phải trả cho sự giảm mạnh trong tham số này
là các đặc trưng của chúng ta bây giờ bất biến dịch chuyển
và lớp của chúng ta chỉ có thể kết hợp thông tin cục bộ,
khi xác định giá trị của mỗi kích hoạt ẩn.
Tất cả việc học phụ thuộc vào việc áp đặt thiên lệch quy nạp.
Khi thiên lệch đó đồng ý với thực tế,
chúng ta có được các mô hình hiệu quả mẫu
tổng quát hóa tốt với dữ liệu chưa thấy.
Nhưng tất nhiên, nếu những thiên lệch đó không đồng ý với thực tế,
ví dụ: nếu hình ảnh hóa ra không bất biến dịch chuyển,
các mô hình của chúng ta có thể khó thậm chí để khớp dữ liệu huấn luyện.

Sự giảm mạnh về tham số này đưa chúng ta đến điều mong muốn cuối cùng,
cụ thể là các lớp sâu hơn nên biểu diễn các khía cạnh lớn hơn và phức tạp hơn
của một hình ảnh. Điều này có thể đạt được bằng cách xen kẽ các phi tuyến tính và các lớp
tích chập lặp đi lặp lại.

## Phép Tích chập

Hãy xem xét ngắn gọn lý do tại sao :eqref:`eq_conv-layer` được gọi là phép tích chập.
Trong toán học, *phép tích chập* giữa hai hàm [Rudin.1973],
giả sử $f, g: \mathbb{R}^d \to \mathbb{R}$ được định nghĩa là

$$(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}.$$

Nghĩa là, chúng ta đo độ chồng lấp giữa $f$ và $g$
khi một hàm bị "lật" và dịch chuyển theo $\mathbf{x}$.
Bất cứ khi nào chúng ta có các đối tượng rời rạc, tích phân trở thành tổng.
Ví dụ, đối với các vectơ từ
tập hợp các vectơ chiều vô hạn có tổng bình phương hội tụ
với chỉ số chạy qua $\mathbb{Z}$ chúng ta thu được định nghĩa sau:

$$(f * g)(i) = \sum_a f(a) g(i-a).$$

Đối với các tensor hai chiều, chúng ta có một tổng tương ứng
với các chỉ số $(a, b)$ cho $f$ và $(i-a, j-b)$ cho $g$, tương ứng:

$$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b).$$

Điều này trông tương tự như :eqref:`eq_conv-layer`, với một sự khác biệt lớn.
Thay vì sử dụng $(i+a, j+b)$, chúng ta sử dụng hiệu thay thế.
Tuy nhiên, lưu ý rằng sự phân biệt này chủ yếu là bề ngoài
vì chúng ta luôn có thể khớp ký hiệu giữa
:eqref:`eq_conv-layer` và :eqref:`eq_2d-conv-discrete`.
Định nghĩa ban đầu của chúng ta trong :eqref:`eq_conv-layer` mô tả chính xác hơn
một *tương quan chéo*.
Chúng ta sẽ quay lại điều này trong phần tiếp theo.


## Kênh
<a id="subsec_why-conv-channels"></a>

Quay lại bộ phát hiện Waldo của chúng ta, hãy xem điều này trông như thế nào.
Lớp tích chập chọn các cửa sổ có kích thước nhất định
và trọng số các cường độ theo bộ lọc $\mathsf{V}$, như được minh họa trong [fig_waldo_mask](#fig_waldo_mask).
Chúng ta có thể nhằm mục đích học một mô hình sao cho
bất cứ nơi nào "độ waldo" cao nhất,
chúng ta nên tìm một đỉnh trong các biểu diễn lớp ẩn.

![Phát hiện Waldo (hình ảnh cảm ơn William Murphy (Infomatique)).](../img/waldo-mask.jpg)
<a id="fig_waldo_mask"></a>

Chỉ có một vấn đề với cách tiếp cận này.
Cho đến nay, chúng ta vui vẻ bỏ qua rằng hình ảnh bao gồm
ba kênh: đỏ, xanh lá, và xanh lam.
Tóm lại, hình ảnh không phải là các đối tượng hai chiều
mà là các tensor bậc ba,
được đặc trưng bởi chiều cao, chiều rộng và kênh,
ví dụ: với hình dạng $1024 \times 1024 \times 3$ điểm ảnh.
Trong khi hai trục đầu tiên trong số này liên quan đến quan hệ không gian,
trục thứ ba có thể được coi là gán
một biểu diễn đa chiều cho mỗi vị trí điểm ảnh.
Do đó chúng ta lập chỉ số $\mathsf{X}$ là $[\mathsf{X}]_{i, j, k}$.
Bộ lọc tích chập phải thích nghi tương ứng.
Thay vì $[\mathbf{V}]_{a,b}$, bây giờ chúng ta có $[\mathsf{V}]_{a,b,c}$.

Hơn nữa, cũng như đầu vào của chúng ta bao gồm một tensor bậc ba,
hóa ra là ý tưởng tốt để tương tự công thức hóa
các biểu diễn ẩn của chúng ta như tensor bậc ba $\mathsf{H}$.
Nói cách khác, thay vì chỉ có một biểu diễn ẩn duy nhất
tương ứng với mỗi vị trí không gian,
chúng ta muốn một vectơ đầy đủ của các biểu diễn ẩn
tương ứng với mỗi vị trí không gian.
Chúng ta có thể nghĩ về các biểu diễn ẩn như bao gồm
một số lưới hai chiều được xếp chồng lên nhau.
Như trong các đầu vào, những thứ này đôi khi được gọi là *kênh*.
Chúng đôi khi còn được gọi là *bản đồ đặc trưng*,
vì mỗi bản đồ cung cấp một tập hợp không gian
các đặc trưng đã học cho lớp tiếp theo.
Một cách trực quan, bạn có thể hình dung rằng ở các lớp thấp hơn gần đầu vào hơn,
một số kênh có thể trở nên chuyên biệt để nhận dạng các cạnh trong khi
các kênh khác có thể nhận dạng kết cấu.

Để hỗ trợ nhiều kênh trong cả đầu vào ($\mathsf{X}$) và biểu diễn ẩn ($\mathsf{H}$),
chúng ta có thể thêm tọa độ thứ tư vào $\mathsf{V}$: $[\mathsf{V}]_{a, b, c, d}$.
Kết hợp tất cả lại chúng ta có:

$$[\mathsf{H}]_{i,j,d} = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c [\mathsf{V}]_{a, b, c, d} [\mathsf{X}]_{i+a, j+b, c},$$

trong đó $d$ lập chỉ số các kênh đầu ra trong biểu diễn ẩn $\mathsf{H}$. Lớp tích chập tiếp theo sẽ tiếp tục nhận một tensor bậc ba, $\mathsf{H}$, làm đầu vào.
Chúng ta lấy
:eqref:`eq_conv-layer-channels`,
vì tính tổng quát của nó, là
định nghĩa của một lớp tích chập cho nhiều kênh, trong đó $\mathsf{V}$ là nhân hoặc bộ lọc của lớp.

Vẫn còn nhiều phép toán mà chúng ta cần giải quyết.
Ví dụ, chúng ta cần tìm ra cách kết hợp tất cả các biểu diễn ẩn
thành một đầu ra duy nhất, ví dụ: liệu có Waldo *ở đâu đó* trong hình ảnh không.
Chúng ta cũng cần quyết định cách tính toán mọi thứ hiệu quả,
cách kết hợp nhiều lớp,
các hàm kích hoạt phù hợp,
và cách đưa ra các lựa chọn thiết kế hợp lý
để tạo ra các mạng hiệu quả trong thực tế.
Chúng ta sẽ chuyển sang những vấn đề này trong phần còn lại của chương.

## Tóm tắt và Thảo luận

Trong phần này, chúng ta đã rút ra cấu trúc của mạng nơ-ron tích chập từ các nguyên tắc đầu tiên. Mặc dù không rõ đây có phải là con đường dẫn đến sự phát minh của CNN hay không, nhưng thật thỏa mãn khi biết rằng chúng là lựa chọn *đúng đắn* khi áp dụng các nguyên tắc hợp lý cho cách thuật toán xử lý hình ảnh và thị giác máy tính nên hoạt động, ít nhất là ở các cấp độ thấp hơn. Đặc biệt, bất biến dịch chuyển trong hình ảnh có nghĩa là tất cả các vùng nhỏ của hình ảnh sẽ được xử lý theo cùng một cách. Cục bộ có nghĩa là chỉ một vùng lân cận nhỏ của các điểm ảnh sẽ được sử dụng để tính toán các biểu diễn ẩn tương ứng. Một số tài liệu tham khảo sớm nhất về CNN là ở dạng Neocognitron [Fukushima.1982].

Nguyên tắc thứ hai mà chúng ta gặp trong lập luận của mình là cách giảm số lượng tham số trong một lớp hàm mà không giới hạn khả năng biểu đạt của nó, ít nhất là bất cứ khi nào một số giả định nhất định về mô hình có giá trị. Chúng ta đã thấy sự giảm mạnh về độ phức tạp là kết quả của hạn chế này, biến các bài toán không khả thi về mặt tính toán và thống kê thành các mô hình có thể xử lý được.

Thêm kênh cho phép chúng ta lấy lại một số độ phức tạp đã bị mất do các hạn chế áp đặt lên nhân tích chập bởi cục bộ và bất biến dịch chuyển. Lưu ý rằng việc thêm các kênh khác ngoài đỏ, xanh lá và xanh lam là khá tự nhiên. Nhiều hình ảnh vệ tinh, đặc biệt cho nông nghiệp và khí tượng học, có từ hàng chục đến hàng trăm kênh, tạo ra các hình ảnh siêu phổ. Chúng báo cáo dữ liệu ở nhiều bước sóng khác nhau. Trong phần tiếp theo chúng ta sẽ thấy cách sử dụng các phép tích chập hiệu quả để thao tác chiều của các hình ảnh mà chúng hoạt động, cách chuyển từ biểu diễn dựa trên vị trí sang biểu diễn dựa trên kênh, và cách xử lý số lượng lớn các danh mục một cách hiệu quả.

## Bài tập

1. Giả sử kích thước của nhân tích chập là $\Delta = 0$.
   Chứng minh rằng trong trường hợp này nhân tích chập
   triển khai một MLP độc lập cho mỗi tập hợp kênh. Điều này dẫn đến
   kiến trúc Network in Network [Lin.Chen.Yan.2013].
1. Dữ liệu âm thanh thường được biểu diễn dưới dạng chuỗi một chiều.
    1. Khi nào bạn muốn áp đặt cục bộ và bất biến dịch chuyển cho âm thanh?
    1. Rút ra các phép toán tích chập cho âm thanh.
    1. Bạn có thể xử lý âm thanh bằng các công cụ tương tự như thị giác máy tính không? Gợi ý: sử dụng phổ đồ.
1. Tại sao bất biến dịch chuyển có thể không phải là ý tưởng tốt sau tất cả? Cho một ví dụ.
1. Bạn có nghĩ rằng các lớp tích chập cũng có thể áp dụng cho dữ liệu văn bản không?
   Bạn có thể gặp phải những vấn đề gì với ngôn ngữ?
1. Điều gì xảy ra với các phép tích chập khi một đối tượng ở ranh giới của hình ảnh?
1. Chứng minh rằng phép tích chập là đối xứng, tức là $f * g = g * f$.

[Discussions](https://discuss.d2l.ai/t/64)
