# Cài Đặt
<a id="chap_installation"></a>

Để có thể bắt đầu chạy,
chúng ta sẽ cần một môi trường để chạy Python,
Jupyter Notebook, các thư viện liên quan,
và code cần thiết để chạy chính cuốn sách này.

## Cài Đặt Miniconda

Lựa chọn đơn giản nhất của bạn là cài đặt
[Miniconda](https://conda.io/en/latest/miniconda.html).
Lưu ý rằng cần phiên bản Python 3.x.
Bạn có thể bỏ qua các bước sau
nếu máy của bạn đã cài conda.

Truy cập trang web Miniconda và xác định
phiên bản phù hợp cho hệ thống của bạn
dựa trên phiên bản Python 3.x và kiến trúc máy.
Giả sử phiên bản Python của bạn là 3.9
(phiên bản chúng tôi đã kiểm thử).
Nếu bạn đang dùng macOS,
bạn sẽ tải script bash
có tên chứa chuỗi "MacOSX",
đi đến vị trí tải xuống,
và thực hiện cài đặt như sau
(lấy Intel Macs làm ví dụ):

```bash
# The file name is subject to changes
sh Miniconda3-py39_4.12.0-MacOSX-x86_64.sh -b
```


Người dùng Linux
sẽ tải file
có tên chứa chuỗi "Linux"
và thực thi lệnh sau tại vị trí tải xuống:

```bash
# The file name is subject to changes
sh Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
```


Người dùng Windows sẽ tải và cài đặt Miniconda bằng cách làm theo [hướng dẫn trực tuyến](https://conda.io/en/latest/miniconda.html) của nó.
Trên Windows, bạn có thể tìm `cmd` để mở Command Prompt (trình thông dịch dòng lệnh) nhằm chạy lệnh.

Tiếp theo, khởi tạo shell để ta có thể chạy `conda` trực tiếp.

```bash
~/miniconda3/bin/conda init
```


Sau đó đóng và mở lại shell hiện tại.
Bạn sẽ có thể tạo
một môi trường mới như sau:

```bash
conda create --name d2l python=3.9 -y
```


Bây giờ ta có thể kích hoạt môi trường `d2l`:

```bash
conda activate d2l
```


## Cài Đặt Framework Deep Learning Và Gói `d2l`

Trước khi cài đặt bất kỳ framework deep learning nào,
trước hết hãy kiểm tra xem
bạn có GPU phù hợp trên máy hay không
(các GPU dùng để hiển thị
trên laptop tiêu chuẩn không liên quan cho mục đích của chúng ta).
Ví dụ,
nếu máy tính của bạn có GPU NVIDIA và đã cài [CUDA](https://developer.nvidia.com/cuda-downloads),
thì bạn đã sẵn sàng.
Nếu máy của bạn không có GPU nào,
hiện tại cũng chưa cần lo lắng.
CPU của bạn cung cấp quá đủ sức mạnh
để bạn đi qua vài chương đầu.
Chỉ cần nhớ rằng bạn sẽ muốn truy cập GPU
trước khi chạy các mô hình lớn hơn.


Bạn có thể cài đặt PyTorch (các phiên bản được chỉ định đã được kiểm thử tại thời điểm viết) với hỗ trợ CPU hoặc GPU như sau:

```bash
pip install torch==2.0.0 torchvision==0.15.1
```


Bước tiếp theo của ta là cài đặt
gói `d2l` mà chúng tôi đã phát triển
để đóng gói
các hàm và lớp thường dùng
xuyên suốt cuốn sách này:

```bash
pip install d2l==1.0.3
```


## Tải Xuống Và Chạy Code

Tiếp theo, bạn sẽ muốn tải xuống các notebook
để có thể chạy từng khối code của cuốn sách.
Chỉ cần nhấp vào tab "Notebooks" ở phía trên
của bất kỳ trang HTML nào trên [trang D2L.ai](https://d2l.ai/)
để tải code xuống rồi giải nén.
Ngoài ra, bạn có thể lấy các notebook
từ dòng lệnh như sau:


```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en-1.0.3.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd pytorch
```


Nếu bạn chưa cài `unzip`, trước hết hãy chạy `sudo apt-get install unzip`.
Bây giờ ta có thể khởi động máy chủ Jupyter Notebook bằng cách chạy:

```bash
jupyter notebook
```


Tại thời điểm này, bạn có thể mở http://localhost:8888
(nó có thể đã tự động mở) trong trình duyệt web của mình.
Sau đó ta có thể chạy code cho từng phần của cuốn sách.
Mỗi khi mở một cửa sổ dòng lệnh mới,
bạn sẽ cần thực thi `conda activate d2l`
để kích hoạt môi trường runtime
trước khi chạy các notebook D2L,
hoặc cập nhật các gói của bạn
(framework deep learning
hoặc gói `d2l`).
Để thoát khỏi môi trường,
chạy `conda deactivate`.


[Thảo luận](https://discuss.d2l.ai/t/24)
