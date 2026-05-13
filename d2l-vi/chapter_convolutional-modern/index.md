# Mạng Nơ-ron Tích chập Hiện đại
<a id="chap_modern_cnn"></a>

Bây giờ chúng ta đã hiểu những kiến thức cơ bản về kết nối các CNN với nhau, hãy cùng
tham quan các kiến trúc CNN hiện đại. Chuyến tham quan này, theo
sự cần thiết, không đầy đủ, nhờ vào vô số thiết kế thú vị mới
đang được thêm vào. Tầm quan trọng của chúng xuất phát từ thực tế là không chỉ
chúng có thể được sử dụng trực tiếp cho các tác vụ thị giác, mà chúng còn đóng vai trò là bộ tạo đặc trưng cơ bản cho các tác vụ nâng cao hơn như theo dõi
[Zhang.Sun.Jiang.ea.2021], phân đoạn [Long.Shelhamer.Darrell.2015], phát hiện đối tượng
[Redmon.Farhadi.2018], hoặc biến đổi phong cách
[Gatys.Ecker.Bethge.2016]. Trong chương này, hầu hết các phần
tương ứng với một kiến trúc CNN quan trọng ở một thời điểm nào đó
(hoặc hiện tại) là mô hình cơ sở mà nhiều dự án nghiên cứu và
hệ thống triển khai được xây dựng. Mỗi mạng này từng là một
kiến trúc thống trị và nhiều mạng là người chiến thắng hoặc á quân trong
[cuộc thi ImageNet](https://www.image-net.org/challenges/LSVRC/)
đã phục vụ như một thước đo tiến bộ trong học có giám sát trong
thị giác máy tính từ năm 2010. Chỉ gần đây Transformer mới bắt đầu
thay thế CNN, bắt đầu với Dosovitskiy.Beyer.Kolesnikov.ea.2021 và
tiếp theo là Swin Transformer [liu2021swin]. Chúng ta sẽ đề cập đến sự phát triển này sau
trong [chap_attention-and-transformers](#chap_attention-and-transformers).

Trong khi ý tưởng về mạng nơ-ron *sâu* khá đơn giản (xếp chồng
một loạt các lớp lại với nhau), hiệu suất có thể thay đổi rất nhiều qua
các lựa chọn kiến trúc và siêu tham số. Các mạng nơ-ron
được mô tả trong chương này là sản phẩm của trực giác, một vài
hiểu biết toán học, và rất nhiều thử nghiệm và sai sót. Chúng ta trình bày những
mô hình này theo thứ tự thời gian, một phần để truyền đạt ý thức về lịch sử
để bạn có thể hình thành trực giác của riêng mình về hướng đi của lĩnh vực
và có thể phát triển kiến trúc của riêng bạn. Ví dụ,
chuẩn hóa batch và kết nối dư được mô tả trong chương này
đã đưa ra hai ý tưởng phổ biến để huấn luyện và thiết kế các mô hình sâu,
cả hai đã được áp dụng cho các kiến trúc ngoài thị giác máy tính.

Chúng ta bắt đầu chuyến tham quan CNN hiện đại với AlexNet [Krizhevsky.Sutskever.Hinton.2012],
mạng quy mô lớn đầu tiên được triển khai để đánh bại các phương pháp thị giác máy tính thông thường
trên một thách thức thị giác quy mô lớn; mạng VGG
[Simonyan.Zisserman.2014], sử dụng một số
khối phần tử lặp đi lặp lại; mạng trong mạng (NiN) tích chập toàn bộ các mạng nơ-ron theo từng vùng qua đầu vào
[Lin.Chen.Yan.2013]; GoogLeNet sử dụng mạng với
tích chập đa nhánh [Szegedy.Liu.Jia.ea.2015]; mạng dư
(ResNet) [He.Zhang.Ren.ea.2016], vẫn là một trong
những kiến trúc phổ biến nhất trong thị giác máy tính;
các khối ResNeXt [Xie.Girshick.Dollar.ea.2017]
cho các kết nối thưa thớt hơn;
và DenseNet
[Huang.Liu.Van-Der-Maaten.ea.2017] cho một tổng quát hóa của
kiến trúc dư. Theo thời gian, nhiều tối ưu hóa đặc biệt cho các mạng hiệu quả
đã được phát triển, chẳng hạn như dịch chuyển tọa độ (ShiftNet) [wu2018shift]. Điều này
đạt đỉnh điểm trong việc tìm kiếm tự động các kiến trúc hiệu quả như
MobileNet v3 [Howard.Sandler.Chu.ea.2019]. Nó cũng bao gồm
khám phá thiết kế bán tự động của Radosavovic.Kosaraju.Girshick.ea.2020
dẫn đến RegNetX/Y mà chúng ta sẽ thảo luận sau trong chương này.
Công trình này mang tính giáo dục ở chỗ nó cung cấp một con đường để kết hợp tính toán sức mạnh thô với
sự thông minh của người thử nghiệm trong việc tìm kiếm các không gian thiết kế hiệu quả. Đáng chú ý là
công trình của liu2022convnet vì nó cho thấy rằng các kỹ thuật huấn luyện (ví dụ: bộ tối ưu hóa, tăng cường dữ liệu và chuẩn hóa)
đóng vai trò then chốt trong việc cải thiện độ chính xác. Nó cũng cho thấy rằng các giả định được giữ lâu dài, chẳng hạn như
kích thước cửa sổ tích chập, có thể cần được xem xét lại, với sự tăng lên của
tính toán và dữ liệu. Chúng ta sẽ đề cập đến điều này và nhiều câu hỏi khác đúng lúc trong chương này.

```toc
:maxdepth: 2

alexnet
vgg
nin
googlenet
batch-norm
resnet
densenet
cnn-design
```
