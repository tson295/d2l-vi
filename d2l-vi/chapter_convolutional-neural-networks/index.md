# Mạng Nơ-ron Tích chập
<a id="chap_cnn"></a>

Dữ liệu hình ảnh được biểu diễn dưới dạng lưới hai chiều của các điểm ảnh,
dù hình ảnh đơn sắc hay màu. Theo đó mỗi điểm ảnh tương ứng với một
hoặc nhiều giá trị số tương ứng. Cho đến nay chúng ta đã bỏ qua cấu trúc phong phú này
và xử lý hình ảnh như các vectơ số bằng cách *làm phẳng* chúng,
bất kể quan hệ không gian giữa các điểm ảnh. Cách tiếp cận không thỏa đáng này
là cần thiết để đưa các vectơ một chiều kết quả
qua một MLP kết nối đầy đủ.

Vì các mạng này bất biến với thứ tự của các đặc trưng, chúng ta
có thể nhận được kết quả tương tự bất kể chúng ta có bảo tồn một thứ tự
tương ứng với cấu trúc không gian của các điểm ảnh hay hoán vị
các cột của ma trận thiết kế trước khi khớp các tham số của MLP.
Lý tưởng nhất, chúng ta muốn tận dụng kiến thức tiên nghiệm rằng các điểm ảnh lân cận
thường có liên quan đến nhau, để xây dựng các mô hình hiệu quả cho
việc học từ dữ liệu hình ảnh.

Chương này giới thiệu *mạng nơ-ron tích chập* (CNN)
[LeCun.Jackel.Bottou.ea.1995], một họ mạng nơ-ron mạnh mẽ được
thiết kế chính xác cho mục đích này.
Các kiến trúc dựa trên CNN hiện đã
phổ biến khắp nơi trong lĩnh vực thị giác máy tính.
Ví dụ, trên bộ sưu tập Imagenet
[Deng.Dong.Socher.ea.2009] chính việc sử dụng mạng nơ-ron tích chập,
viết tắt là Convnet, đã mang lại những cải thiện hiệu suất đáng kể
[Krizhevsky.Sutskever.Hinton.2012].

Các CNN hiện đại, như chúng thường được gọi, có thiết kế lấy cảm hứng từ
sinh học, lý thuyết nhóm, và nhiều thử nghiệm thực tế. Ngoài hiệu quả mẫu trong
việc đạt được các mô hình chính xác, CNN có xu hướng hiệu quả về mặt tính toán,
cả vì chúng yêu cầu ít tham số hơn các kiến trúc kết nối đầy đủ
và vì các phép tích chập dễ dàng song song hóa trên các lõi GPU
[Chetlur.Woolley.Vandermersch.ea.2014]. Do đó, các nhà thực hành thường
áp dụng CNN bất cứ khi nào có thể, và ngày càng chúng đã nổi lên như
các đối thủ đáng tin cậy ngay cả trên các tác vụ có cấu trúc chuỗi một chiều,
chẳng hạn như âm thanh [Abdel-Hamid.Mohamed.Jiang.ea.2014], văn bản
[Kalchbrenner.Grefenstette.Blunsom.2014], và phân tích chuỗi thời gian
[LeCun.Bengio.ea.1995], nơi mạng nơ-ron hồi quy thường được sử dụng.
Một số điều chỉnh thông minh của CNN cũng đã
đưa chúng vào xử lý dữ liệu cấu trúc đồ thị [Kipf.Welling.2016] và
trong hệ thống gợi ý.

Đầu tiên, chúng ta sẽ đi sâu hơn vào động lực của mạng nơ-ron tích chập.
Tiếp theo là tổng quan về các phép toán cơ bản
tạo nên xương sống của tất cả các mạng tích chập.
Những điều này bao gồm bản thân các lớp tích chập,
các chi tiết quan trọng bao gồm đệm và sải bước,
các lớp gộp được sử dụng để tổng hợp thông tin
trên các vùng không gian lân cận,
việc sử dụng nhiều kênh ở mỗi lớp,
và thảo luận cẩn thận về cấu trúc của các kiến trúc hiện đại.
Chúng ta sẽ kết thúc chương với một ví dụ hoạt động đầy đủ về LeNet,
mạng tích chập đầu tiên được triển khai thành công,
từ lâu trước khi sự trỗi dậy của deep learning hiện đại.
Trong chương tiếp theo, chúng ta sẽ đi vào triển khai đầy đủ
của một số kiến trúc CNN phổ biến và tương đối gần đây
mà thiết kế của chúng đại diện cho hầu hết các kỹ thuật
thường được sử dụng bởi các nhà thực hành hiện đại.

```toc
:maxdepth: 2

why-conv
conv-layer
padding-and-strides
channels
pooling
lenet
```
