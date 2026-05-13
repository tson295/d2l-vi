# Tổng quát hóa trong Deep Learning


Trong [chap_regression](#chap_regression) và [chap_classification](#chap_classification),
chúng ta đã giải quyết các bài toán hồi quy và phân loại
bằng cách khớp các mô hình tuyến tính với dữ liệu huấn luyện.
Trong cả hai trường hợp, chúng ta cung cấp các thuật toán thực tế
để tìm các tham số tối đa hóa
khả năng xảy ra của các nhãn huấn luyện quan sát được.
Và sau đó, về cuối mỗi chương,
chúng ta nhớ lại rằng việc khớp dữ liệu huấn luyện
chỉ là một mục tiêu trung gian.
Mục tiêu thực sự của chúng ta là khám phá *các mẫu tổng quát*
dựa vào đó chúng ta có thể đưa ra dự đoán chính xác
ngay cả trên các ví dụ mới được lấy từ cùng một tổng thể cơ sở.
Các nhà nghiên cứu machine learning là *người dùng* của các thuật toán tối ưu hóa.
Đôi khi, chúng ta thậm chí phải phát triển các thuật toán tối ưu hóa mới.
Nhưng cuối cùng, tối ưu hóa chỉ là phương tiện để đạt mục đích.
Về cốt lõi, machine learning là một ngành thống kê
và chúng ta muốn tối ưu hóa mất mát huấn luyện chỉ khi
một nguyên tắc thống kê nào đó (được biết hoặc không được biết)
dẫn đến các mô hình kết quả tổng quát hóa vượt ra ngoài tập huấn luyện.


Mặt tươi sáng là, hóa ra các mạng nơ-ron sâu
được huấn luyện bởi gradient descent ngẫu nhiên tổng quát hóa đáng kể tốt
trên vô số bài toán dự đoán, bao gồm thị giác máy tính;
xử lý ngôn ngữ tự nhiên; dữ liệu chuỗi thời gian; hệ thống gợi ý;
hồ sơ sức khỏe điện tử; gấp protein;
ước tính hàm giá trị trong trò chơi điện tử
và trò chơi bảng; và nhiều lĩnh vực khác.
Mặt không tươi sáng là, nếu bạn đang tìm kiếm
một giải thích đơn giản
về câu chuyện tối ưu hóa
(tại sao chúng ta có thể khớp chúng với dữ liệu huấn luyện)
hoặc câu chuyện tổng quát hóa
(tại sao các mô hình kết quả tổng quát hóa cho các ví dụ chưa thấy),
thì bạn có thể muốn rót cho mình một ly đồ uống.
Trong khi các quy trình tối ưu hóa mô hình tuyến tính
và các đặc tính thống kê của nghiệm
đều được mô tả tốt bởi một hệ thống lý thuyết toàn diện,
sự hiểu biết của chúng ta về deep learning
vẫn giống như miền Tây hoang dã trên cả hai mặt trận.

Cả lý thuyết và thực hành của deep learning
đang phát triển nhanh chóng,
với các lý thuyết gia áp dụng các chiến lược mới
để giải thích những gì đang xảy ra,
trong khi các nhà thực hành tiếp tục
đổi mới với tốc độ chóng mặt,
xây dựng kho vũ khí heuristic để huấn luyện mạng sâu
và một tập hợp trực giác và kiến thức dân gian
cung cấp hướng dẫn để quyết định
kỹ thuật nào áp dụng trong tình huống nào.

Tóm tắt thời điểm hiện tại là lý thuyết deep learning
đã tạo ra các hướng tấn công đầy hứa hẹn và các kết quả thú vị rải rác,
nhưng vẫn còn xa so với một giải trình toàn diện
về cả (i) tại sao chúng ta có thể tối ưu hóa mạng nơ-ron
và (ii) làm thế nào các mô hình được học bởi gradient descent
quản lý để tổng quát hóa tốt như vậy, ngay cả trên các tác vụ nhiều chiều.
Tuy nhiên, trong thực tế, (i) hiếm khi là vấn đề
(chúng ta luôn có thể tìm thấy các tham số sẽ khớp tất cả dữ liệu huấn luyện của chúng ta)
và do đó hiểu sự tổng quát hóa là vấn đề lớn hơn nhiều.
Mặt khác, ngay cả khi không có sự an ủi của một lý thuyết khoa học mạch lạc,
các nhà thực hành đã phát triển một tập hợp lớn các kỹ thuật
có thể giúp bạn tạo ra các mô hình tổng quát hóa tốt trong thực tế.
Mặc dù không có tóm tắt ngắn gọn nào có thể làm công bằng
với chủ đề rộng lớn về tổng quát hóa trong deep learning,
và trong khi trạng thái tổng thể của nghiên cứu còn chưa được giải quyết,
chúng ta hy vọng, trong phần này, trình bày một cái nhìn tổng quan rộng
về trạng thái nghiên cứu và thực hành.


## Xem lại Quá khớp và Chuẩn hóa

Theo định lý "không có bữa ăn miễn phí" của wolpert1995no,
bất kỳ thuật toán học nào cũng tổng quát hóa tốt hơn trên dữ liệu với một số phân phối nhất định, và tệ hơn với các phân phối khác.
Do đó, với một tập huấn luyện hữu hạn,
một mô hình dựa vào một số giả định nhất định:
để đạt hiệu suất cấp độ con người
có thể hữu ích khi xác định *thiên kiến quy nạp*
phản ánh cách con người nghĩ về thế giới.
Những thiên kiến quy nạp như vậy thể hiện sở thích
cho các nghiệm có các đặc tính nhất định.
Ví dụ,
một MLP sâu có thiên kiến quy nạp
hướng tới việc xây dựng một hàm phức tạp bằng cách kết hợp các hàm đơn giản hơn.

Với các mô hình machine learning mã hóa các thiên kiến quy nạp,
cách tiếp cận của chúng ta để huấn luyện chúng
thường bao gồm hai giai đoạn: (i) khớp dữ liệu huấn luyện;
và (ii) ước tính *lỗi tổng quát hóa*
(lỗi thực sự trên tổng thể cơ sở)
bằng cách đánh giá mô hình trên dữ liệu holdout.
Sự chênh lệch giữa độ khớp trên dữ liệu huấn luyện
và độ khớp trên dữ liệu kiểm tra được gọi là *khoảng cách tổng quát hóa* và khi điều này lớn,
chúng ta nói rằng các mô hình của chúng ta *quá khớp* với dữ liệu huấn luyện.
Trong các trường hợp cực đoan của quá khớp,
chúng ta có thể khớp chính xác dữ liệu huấn luyện,
ngay cả khi lỗi kiểm tra vẫn còn đáng kể.
Và trong quan điểm cổ điển,
sự giải thích là các mô hình của chúng ta quá phức tạp,
đòi hỏi chúng ta phải thu nhỏ số lượng đặc trưng,
số lượng tham số khác không được học,
hoặc kích thước của các tham số như được định lượng.
Hãy nhớ lại biểu đồ về độ phức tạp của mô hình so với mất mát
([fig_capacity_vs_error](#fig_capacity_vs_error))
từ [sec_generalization_basics](#sec_generalization_basics).


Tuy nhiên deep learning làm phức tạp bức tranh này theo những cách phản trực giác.
Đầu tiên, đối với các bài toán phân loại,
các mô hình của chúng ta thường đủ biểu diễn
để khớp hoàn hảo mọi ví dụ huấn luyện,
ngay cả trong các tập dữ liệu bao gồm hàng triệu
[zhang2021understanding].
Trong bức tranh cổ điển, chúng ta có thể nghĩ
rằng cài đặt này nằm ở cực bên phải xa
của trục độ phức tạp mô hình,
và rằng bất kỳ cải thiện nào trong lỗi tổng quát hóa
phải đến thông qua chuẩn hóa,
bằng cách giảm độ phức tạp của lớp mô hình,
hoặc bằng cách áp dụng một hình phạt, ràng buộc nghiêm ngặt
tập hợp các giá trị mà các tham số của chúng ta có thể nhận.
Nhưng đó là lúc mọi thứ bắt đầu trở nên kỳ lạ.

Thật kỳ lạ, đối với nhiều tác vụ deep learning
(ví dụ: nhận dạng hình ảnh và phân loại văn bản)
chúng ta thường đang chọn trong số các kiến trúc mô hình,
tất cả đều có thể đạt được mất mát huấn luyện tùy ý thấp
(và lỗi huấn luyện bằng không).
Vì tất cả các mô hình được xem xét đều đạt lỗi huấn luyện bằng không,
*con đường duy nhất để đạt được lợi ích thêm là giảm quá khớp*.
Thậm chí kỳ lạ hơn, thường xảy ra trường hợp
mặc dù khớp hoàn hảo với dữ liệu huấn luyện,
chúng ta thực sự có thể *giảm thêm lỗi tổng quát hóa*
bằng cách làm cho mô hình *thậm chí biểu diễn hơn*,
ví dụ: thêm lớp, nút, hoặc huấn luyện
trong một số epoch lớn hơn.
Kỳ lạ hơn nữa, mẫu liên quan đến khoảng cách tổng quát hóa
với *độ phức tạp* của mô hình (như được nắm bắt, ví dụ, trong độ sâu hoặc chiều rộng của mạng)
có thể không đơn điệu,
với độ phức tạp lớn hơn gây hại ban đầu
nhưng sau đó lại giúp ích trong cái gọi là mẫu "double-descent"
[nakkiran2021deep].
Do đó, người thực hành deep learning sở hữu một túi thủ thuật,
một số trong đó dường như hạn chế mô hình theo một cách nào đó
và một số khác dường như làm cho nó thậm chí biểu diễn hơn,
và tất cả đều, theo một nghĩa nào đó, được áp dụng để giảm thiểu quá khớp.

Làm phức tạp thêm mọi thứ,
trong khi các đảm bảo được cung cấp bởi lý thuyết học cổ điển
có thể bảo thủ ngay cả đối với các mô hình cổ điển,
chúng có vẻ bất lực để giải thích tại sao
các mạng nơ-ron sâu tổng quát hóa ngay từ đầu.
Vì các mạng nơ-ron sâu có khả năng khớp
các nhãn tùy ý ngay cả đối với các tập dữ liệu lớn,
và mặc dù sử dụng các phương pháp quen thuộc như chuẩn hóa $\ell_2$,
các giới hạn tổng quát hóa dựa trên độ phức tạp truyền thống,
ví dụ: dựa trên chiều VC
hoặc độ phức tạp Rademacher của một lớp giả thuyết
không thể giải thích tại sao mạng nơ-ron tổng quát hóa.

## Cảm hứng từ Phi tham số

Khi tiếp cận deep learning lần đầu tiên,
thật hấp dẫn khi nghĩ đến chúng như là các mô hình tham số.
Xét cho cùng, các mô hình *có* hàng triệu tham số.
Khi chúng ta cập nhật các mô hình, chúng ta cập nhật các tham số của chúng.
Khi chúng ta lưu các mô hình, chúng ta ghi các tham số của chúng ra đĩa.
Tuy nhiên, toán học và khoa học máy tính chứa đầy
những thay đổi phản trực giác về quan điểm,
và những đẳng cấu đáng ngạc nhiên giữa các bài toán dường như khác nhau.
Mặc dù mạng nơ-ron rõ ràng *có* tham số,
theo một số cách, có thể hữu ích hơn
khi nghĩ về chúng như là hoạt động giống mô hình phi tham số.
Vậy chính xác điều gì làm cho một mô hình phi tham số?
Mặc dù tên này bao gồm một tập hợp đa dạng các phương pháp,
một chủ đề chung là các phương pháp phi tham số
có xu hướng có mức độ phức tạp tăng
khi lượng dữ liệu có sẵn tăng.

Có lẽ ví dụ đơn giản nhất về một mô hình phi tham số
là thuật toán $k$-láng giềng gần nhất (chúng ta sẽ đề cập đến nhiều mô hình phi tham số hơn sau, ví dụ trong [sec_attention-pooling](#sec_attention-pooling)).
Ở đây, trong thời gian huấn luyện,
người học đơn giản ghi nhớ tập dữ liệu.
Sau đó, trong thời gian dự đoán,
khi đối mặt với một điểm mới $\mathbf{x}$,
người học tra cứu $k$ láng giềng gần nhất
($k$ điểm $\mathbf{x}_i'$ tối thiểu hóa
một khoảng cách nào đó $d(\mathbf{x}, \mathbf{x}_i')$).
Khi $k=1$, thuật toán này được gọi là $1$-láng giềng gần nhất,
và thuật toán sẽ luôn đạt lỗi huấn luyện bằng không.
Tuy nhiên, điều đó không có nghĩa là thuật toán sẽ không tổng quát hóa.
Trên thực tế, hóa ra dưới một số điều kiện nhẹ,
thuật toán 1-láng giềng gần nhất là nhất quán
(cuối cùng hội tụ về bộ dự đoán tối ưu).


Lưu ý rằng $1$-láng giềng gần nhất yêu cầu chúng ta chỉ định
một hàm khoảng cách $d$ nào đó, hoặc tương đương,
rằng chúng ta chỉ định một hàm cơ sở có giá trị vectơ $\phi(\mathbf{x})$
để rút đặc trưng dữ liệu của chúng ta.
Đối với bất kỳ lựa chọn số liệu khoảng cách nào,
chúng ta sẽ đạt lỗi huấn luyện bằng không
và cuối cùng đạt đến bộ dự đoán tối ưu,
nhưng các số liệu khoảng cách khác nhau $d$
mã hóa các thiên kiến quy nạp khác nhau
và với lượng dữ liệu có sẵn hữu hạn
sẽ mang lại các bộ dự đoán khác nhau.
Các lựa chọn khác nhau của số liệu khoảng cách $d$
đại diện cho các giả định khác nhau về các mẫu cơ sở
và hiệu suất của các bộ dự đoán khác nhau
sẽ phụ thuộc vào mức độ tương thích của các giả định
với dữ liệu quan sát được.

Theo một nghĩa nào đó, vì mạng nơ-ron được tham số hóa quá mức,
sở hữu nhiều tham số hơn mức cần thiết để khớp dữ liệu huấn luyện,
chúng có xu hướng *nội suy* dữ liệu huấn luyện (khớp hoàn hảo)
và do đó hoạt động, theo một số cách, giống mô hình phi tham số hơn.
Nghiên cứu lý thuyết gần đây hơn đã thiết lập
mối liên hệ sâu sắc giữa các mạng nơ-ron lớn
và các phương pháp phi tham số, đặc biệt là phương pháp kernel.
Cụ thể, Jacot.Grabriel.Hongler.2018
đã chứng minh rằng ở giới hạn, khi perceptron đa lớp
với các trọng số được khởi tạo ngẫu nhiên trở nên rộng vô hạn,
chúng trở nên tương đương với các phương pháp kernel (phi tham số)
đối với một lựa chọn cụ thể của hàm kernel
(về cơ bản là một hàm khoảng cách),
mà họ gọi là kernel tangent mạng nơ-ron.
Mặc dù các mô hình kernel tangent mạng nơ-ron hiện tại có thể không giải thích đầy đủ
hành vi của các mạng sâu hiện đại,
sự thành công của chúng như một công cụ phân tích
nhấn mạnh tính hữu ích của mô hình hóa phi tham số
để hiểu hành vi của các mạng sâu được tham số hóa quá mức.


## Dừng Sớm

Mặc dù các mạng nơ-ron sâu có khả năng khớp các nhãn tùy ý,
ngay cả khi nhãn được gán sai hoặc ngẫu nhiên
[zhang2021understanding],
khả năng này chỉ xuất hiện sau nhiều lần lặp huấn luyện.
Một hướng nghiên cứu mới [Rolnick.Veit.Belongie.Shavit.2017]
đã tiết lộ rằng trong cài đặt nhiễu nhãn,
mạng nơ-ron có xu hướng khớp dữ liệu được gán nhãn sạch trước
và chỉ sau đó mới nội suy dữ liệu bị gán nhãn sai.
Hơn nữa, đã được thiết lập rằng hiện tượng này
chuyển hóa trực tiếp thành đảm bảo về tổng quát hóa:
bất cứ khi nào một mô hình đã khớp dữ liệu được gán nhãn sạch
nhưng không phải các ví dụ được gán nhãn ngẫu nhiên được đưa vào tập huấn luyện,
trên thực tế nó đã tổng quát hóa [Garg.Balakrishnan.Kolter.Lipton.2021].

Cùng nhau, những phát hiện này giúp thúc đẩy *dừng sớm*,
một kỹ thuật cổ điển để chuẩn hóa các mạng nơ-ron sâu.
Ở đây, thay vì trực tiếp ràng buộc các giá trị của trọng số,
người ta ràng buộc số epoch huấn luyện.
Cách phổ biến nhất để xác định tiêu chí dừng
là theo dõi lỗi kiểm định trong suốt quá trình huấn luyện
(thường bằng cách kiểm tra một lần sau mỗi epoch)
và cắt bỏ huấn luyện khi lỗi kiểm định
không giảm hơn một lượng nhỏ $\epsilon$
trong một số epoch.
Đây đôi khi được gọi là *tiêu chí kiên nhẫn*.
Ngoài khả năng dẫn đến tổng quát hóa tốt hơn
trong cài đặt nhãn nhiễu,
một lợi ích khác của dừng sớm là thời gian tiết kiệm được.
Một khi tiêu chí kiên nhẫn được đáp ứng, người ta có thể kết thúc huấn luyện.
Đối với các mô hình lớn có thể yêu cầu nhiều ngày huấn luyện
đồng thời trên tám hoặc nhiều GPU hơn,
việc dừng sớm được điều chỉnh tốt có thể tiết kiệm nhiều ngày thời gian cho các nhà nghiên cứu
và có thể tiết kiệm nhiều nghìn đô la cho chủ lao động của họ.

Đáng chú ý, khi không có nhiễu nhãn và các tập dữ liệu là *có thể thực hiện được*
(các lớp thực sự có thể tách biệt, ví dụ: phân biệt mèo với chó),
dừng sớm có xu hướng không dẫn đến cải thiện đáng kể trong tổng quát hóa.
Mặt khác, khi có nhiễu nhãn,
hoặc tính biến đổi vốn có trong nhãn
(ví dụ: dự đoán tỷ lệ tử vong của bệnh nhân),
dừng sớm là cực kỳ quan trọng.
Huấn luyện các mô hình cho đến khi chúng nội suy dữ liệu nhiễu thường là ý tưởng tồi.


## Các Phương pháp Chuẩn hóa Cổ điển cho Mạng Sâu

Trong [chap_regression](#chap_regression), chúng ta đã mô tả
một số kỹ thuật chuẩn hóa cổ điển
để ràng buộc độ phức tạp của các mô hình.
Cụ thể, [sec_weight_decay](#sec_weight_decay)
giới thiệu một phương pháp gọi là suy giảm trọng số (weight decay),
bao gồm việc thêm một số hạng chuẩn hóa vào hàm mất mát
để phạt các giá trị lớn của trọng số.
Tùy thuộc vào chuẩn trọng số nào bị phạt
kỹ thuật này được gọi là chuẩn hóa ridge (đối với hình phạt $\ell_2$)
hoặc chuẩn hóa lasso (đối với hình phạt $\ell_1$).
Trong phân tích cổ điển về các bộ chuẩn hóa này,
chúng được coi là đủ hạn chế về các giá trị
mà trọng số có thể nhận để ngăn mô hình khớp các nhãn tùy ý.

Trong các cài đặt deep learning,
suy giảm trọng số vẫn là một công cụ phổ biến.
Tuy nhiên, các nhà nghiên cứu đã nhận thấy
rằng cường độ điển hình của chuẩn hóa $\ell_2$
không đủ để ngăn các mạng
nội suy dữ liệu [zhang2021understanding] và do đó lợi ích nếu được giải thích
như chuẩn hóa chỉ có thể có ý nghĩa
kết hợp với tiêu chí dừng sớm.
Vắng mặt dừng sớm, có thể
rằng giống như số lớp
hoặc số nút (trong deep learning)
hoặc số liệu khoảng cách (trong 1-láng giềng gần nhất),
các phương pháp này có thể dẫn đến tổng quát hóa tốt hơn
không phải vì chúng ràng buộc một cách có ý nghĩa
sức mạnh của mạng nơ-ron
mà thay vào đó vì bằng cách nào đó chúng mã hóa các thiên kiến quy nạp
tương thích tốt hơn với các mẫu
được tìm thấy trong các tập dữ liệu quan tâm.
Do đó, các bộ chuẩn hóa cổ điển vẫn phổ biến
trong các cài đặt deep learning,
mặc dù cơ sở lý thuyết
cho hiệu quả của chúng có thể khác biệt hoàn toàn.

Đáng chú ý, các nhà nghiên cứu deep learning cũng đã xây dựng
dựa trên các kỹ thuật được phổ biến lần đầu tiên
trong các bối cảnh chuẩn hóa cổ điển,
chẳng hạn như thêm nhiễu vào đầu vào mô hình.
Trong phần tiếp theo chúng ta sẽ giới thiệu
kỹ thuật dropout nổi tiếng
(được phát minh bởi Srivastava.Hinton.Krizhevsky.ea.2014),
đã trở thành một trụ cột của deep learning,
mặc dù cơ sở lý thuyết cho hiệu quả của nó
vẫn còn bí ẩn tương tự.


## Tóm tắt

Không giống như các mô hình tuyến tính cổ điển,
có xu hướng có ít tham số hơn ví dụ,
các mạng sâu có xu hướng được tham số hóa quá mức,
và đối với hầu hết các tác vụ có khả năng
khớp hoàn hảo tập huấn luyện.
*Chế độ nội suy* này thách thức
nhiều trực giác vững chắc được giữ lâu nay.
Về mặt chức năng, mạng nơ-ron trông giống mô hình tham số.
Nhưng việc nghĩ đến chúng như mô hình phi tham số
đôi khi có thể là nguồn trực giác đáng tin cậy hơn.
Vì thường xảy ra trường hợp tất cả mạng sâu được xem xét
đều có khả năng khớp tất cả nhãn huấn luyện,
hầu hết tất cả lợi ích phải đến từ việc giảm thiểu quá khớp
(thu hẹp *khoảng cách tổng quát hóa*).
Nghịch lý thay, các can thiệp
làm giảm khoảng cách tổng quát hóa
đôi khi dường như làm tăng độ phức tạp mô hình
và đôi khi dường như làm giảm độ phức tạp.
Tuy nhiên, các phương pháp này hiếm khi giảm độ phức tạp
đủ để lý thuyết cổ điển
giải thích sự tổng quát hóa của mạng sâu,
và *tại sao một số lựa chọn nhất định dẫn đến tổng quát hóa cải thiện*
phần lớn vẫn là một câu hỏi mở rộng lớn
mặc dù nỗ lực tập trung của nhiều nhà nghiên cứu xuất sắc.


## Bài tập

1. Theo nghĩa nào các phép đo độ phức tạp truyền thống không tính đến sự tổng quát hóa của mạng nơ-ron sâu?
1. Tại sao *dừng sớm* có thể được coi là kỹ thuật chuẩn hóa?
1. Các nhà nghiên cứu thường xác định tiêu chí dừng như thế nào?
1. Yếu tố quan trọng nào dường như phân biệt các trường hợp khi dừng sớm dẫn đến cải thiện lớn trong tổng quát hóa?
1. Ngoài tổng quát hóa, hãy mô tả một lợi ích khác của dừng sớm.

[Discussions](https://discuss.d2l.ai/t/7473)
