# Kiến Trúc Bộ Mã Hóa--Bộ Giải Mã
<a id="sec_encoder-decoder"></a>

Trong các bài toán chuỗi-sang-chuỗi tổng quát
như dịch máy
([sec_machine_translation](#sec_machine_translation)),
đầu vào và đầu ra có độ dài thay đổi
và không được căn chỉnh.
Cách tiếp cận tiêu chuẩn để xử lý loại dữ liệu này
là thiết kế một kiến trúc *bộ mã hóa--bộ giải mã* ([fig_encoder_decoder](#fig_encoder_decoder))
gồm hai thành phần chính:
một *bộ mã hóa* nhận một chuỗi có độ dài thay đổi làm đầu vào,
và một *bộ giải mã* hoạt động như mô hình ngôn ngữ có điều kiện,
nhận đầu vào đã mã hóa
và ngữ cảnh bên trái của chuỗi đích
và dự đoán token tiếp theo trong chuỗi đích.


![Kiến trúc bộ mã hóa--bộ giải mã.](../img/encoder-decoder.svg)
<a id="fig_encoder_decoder"></a>

Hãy lấy dịch máy từ tiếng Anh sang tiếng Pháp làm ví dụ.
Cho một chuỗi đầu vào bằng tiếng Anh:
"They", "are", "watching", ".",
kiến trúc bộ mã hóa--bộ giải mã này
đầu tiên mã hóa đầu vào có độ dài thay đổi thành một trạng thái,
sau đó giải mã trạng thái
để tạo chuỗi đã dịch,
token theo token, làm đầu ra:
"Ils", "regardent", ".".
Vì kiến trúc bộ mã hóa--bộ giải mã
tạo thành nền tảng của các mô hình chuỗi-sang-chuỗi khác nhau
trong các phần tiếp theo,
phần này sẽ chuyển đổi kiến trúc này
thành một giao diện sẽ được lập trình sau.


```python
from d2l import torch as d2l
from torch import nn
```


## (**Bộ Mã Hóa**)

Trong giao diện bộ mã hóa,
chúng ta chỉ chỉ định rằng
bộ mã hóa nhận các chuỗi có độ dài thay đổi làm đầu vào `X`.
Lập trình sẽ được cung cấp
bởi bất kỳ mô hình nào kế thừa lớp `Encoder` cơ sở này.


```python
class Encoder(nn.Module):  
    """The base encoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError
```


## [**Bộ Giải Mã**]

Trong giao diện bộ giải mã sau,
chúng ta thêm một phương thức `init_state` bổ sung
để chuyển đổi đầu ra bộ mã hóa (`enc_all_outputs`)
thành trạng thái đã mã hóa.
Lưu ý rằng bước này
có thể yêu cầu thêm đầu vào,
chẳng hạn như độ dài hợp lệ của đầu vào,
được giải thích
trong [sec_machine_translation](#sec_machine_translation).
Để tạo ra chuỗi có độ dài thay đổi token theo token,
mỗi lần bộ giải mã có thể ánh xạ một đầu vào
(ví dụ: token được tạo tại bước thời gian trước)
và trạng thái đã mã hóa
thành một token đầu ra tại bước thời gian hiện tại.


```python
class Decoder(nn.Module):  
    """The base decoder interface for the encoder--decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_all_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```


## [**Kết Hợp Bộ Mã Hóa và Bộ Giải Mã**]

Trong lan truyền tiến,
đầu ra của bộ mã hóa
được sử dụng để tạo ra trạng thái đã mã hóa,
và trạng thái này sẽ được sử dụng thêm
bởi bộ giải mã như một trong các đầu vào của nó.


Trong phần tiếp theo,
chúng ta sẽ xem cách áp dụng RNN để thiết kế
các mô hình chuỗi-sang-chuỗi dựa trên
kiến trúc bộ mã hóa--bộ giải mã này.


## Tóm Tắt

Các kiến trúc bộ mã hóa-bộ giải mã
có thể xử lý đầu vào và đầu ra
đều gồm các chuỗi có độ dài thay đổi
và do đó phù hợp với các bài toán chuỗi-sang-chuỗi
như dịch máy.
Bộ mã hóa nhận một chuỗi có độ dài thay đổi làm đầu vào
và biến đổi nó thành một trạng thái có hình dạng cố định.
Bộ giải mã ánh xạ trạng thái đã mã hóa có hình dạng cố định
thành một chuỗi có độ dài thay đổi.


## Bài Tập

1. Giả sử rằng chúng ta sử dụng mạng nơ-ron để lập trình kiến trúc bộ mã hóa--bộ giải mã. Bộ mã hóa và bộ giải mã có phải là cùng loại mạng nơ-ron không?
1. Ngoài dịch máy, bạn có thể nghĩ đến ứng dụng nào khác mà kiến trúc bộ mã hóa--bộ giải mã có thể được áp dụng không?


[Thảo luận](https://discuss.d2l.ai/t/1061)
