# Word Embedding Với Global Vectors (GloVe)
<a id="sec_glove"></a>


Các đồng xuất hiện giữa từ với từ
trong cửa sổ ngữ cảnh
có thể mang thông tin ngữ nghĩa phong phú.
Ví dụ,
trong một kho ngữ liệu lớn,
từ "solid"
có nhiều khả năng đồng xuất hiện
với "ice" hơn "steam",
nhưng từ "gas"
có lẽ đồng xuất hiện với "steam"
thường xuyên hơn "ice".
Bên cạnh đó,
thống kê toàn cục của kho ngữ liệu
về các đồng xuất hiện như vậy
có thể được tính trước:
điều này có thể dẫn tới huấn luyện hiệu quả hơn.
Để tận dụng thông tin thống kê
trong toàn bộ kho ngữ liệu
cho word embedding,
trước hết hãy xem lại
mô hình skip-gram trong [subsec_skip-gram](#subsec_skip-gram),
nhưng diễn giải nó
bằng các thống kê toàn cục của kho ngữ liệu
như số lần đồng xuất hiện.

## Skip-Gram Với Thống Kê Toàn Cục Của Kho Ngữ Liệu
<a id="subsec_skipgram-global"></a>

Ký hiệu $q_{ij}$
là xác suất có điều kiện
$P(w_j\mid w_i)$
của từ $w_j$ khi biết từ $w_i$
trong mô hình skip-gram,
ta có

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \exp(\mathbf{u}_k^\top \mathbf{v}_i)},$$

trong đó
với bất kỳ chỉ số $i$ nào,
các vector $\mathbf{v}_i$ và $\mathbf{u}_i$
biểu diễn từ $w_i$
lần lượt như từ trung tâm và từ ngữ cảnh, và $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$
là tập chỉ số của từ vựng.

Xét từ $w_i$
có thể xuất hiện nhiều lần
trong kho ngữ liệu.
Trong toàn bộ kho ngữ liệu,
tất cả các từ ngữ cảnh
ở bất kỳ nơi nào $w_i$ được lấy làm từ trung tâm
tạo thành một *multiset* $\mathcal{C}_i$
các chỉ số từ,
*cho phép nhiều bản sao của cùng một phần tử*.
Với bất kỳ phần tử nào,
số bản sao của nó được gọi là *bội số*.
Để minh họa bằng một ví dụ,
giả sử từ $w_i$ xuất hiện hai lần trong kho ngữ liệu
và chỉ số của các từ ngữ cảnh
lấy $w_i$ làm từ trung tâm
trong hai cửa sổ ngữ cảnh
là
$k, j, m, k$ và $k, l, k, j$.
Do đó, multiset $\mathcal{C}_i = \{j, j, k, k, k, k, l, m\}$, trong đó
bội số của các phần tử $j, k, l, m$
lần lượt là 2, 4, 1, 1.

Bây giờ hãy ký hiệu bội số của phần tử $j$ trong
multiset $\mathcal{C}_i$ là $x_{ij}$.
Đây là số lần đồng xuất hiện toàn cục
của từ $w_j$ (làm từ ngữ cảnh)
và từ $w_i$ (làm từ trung tâm)
trong cùng cửa sổ ngữ cảnh
trên toàn bộ kho ngữ liệu.
Dùng các thống kê toàn cục của kho ngữ liệu như vậy,
hàm mất mát của mô hình skip-gram
tương đương với

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$

Ta ký hiệu thêm
$x_i$
là số lượng tất cả các từ ngữ cảnh
trong các cửa sổ ngữ cảnh
nơi $w_i$ xuất hiện làm từ trung tâm,
tương đương với $|\mathcal{C}_i|$.
Đặt $p_{ij}$
là xác suất có điều kiện
$x_{ij}/x_i$ để sinh
từ ngữ cảnh $w_j$ khi biết từ trung tâm $w_i$,
:eqref:`eq_skipgram-x_ij`
có thể được viết lại thành

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$

Trong :eqref:`eq_skipgram-p_ij`, $-\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$ tính
cross-entropy
của
phân bố có điều kiện $p_{ij}$
từ thống kê toàn cục của kho ngữ liệu
và
phân bố có điều kiện $q_{ij}$
của dự đoán mô hình.
Mất mát này
cũng được gán trọng số bởi $x_i$ như đã giải thích ở trên.
Tối thiểu hóa hàm mất mát trong
:eqref:`eq_skipgram-p_ij`
sẽ cho phép
phân bố có điều kiện dự đoán
tiệm cận
phân bố có điều kiện
từ thống kê toàn cục của kho ngữ liệu.


Mặc dù thường được dùng
để đo khoảng cách
giữa các phân bố xác suất,
hàm mất mát cross-entropy có thể không phải là lựa chọn tốt ở đây.
Một mặt, như đã đề cập trong [sec_approx_train](#sec_approx_train),
chi phí chuẩn hóa đúng $q_{ij}$
dẫn tới phép lấy tổng trên toàn bộ từ vựng,
có thể tốn kém về tính toán.
Mặt khác,
một số lượng lớn các sự kiện hiếm
từ một kho ngữ liệu lớn
thường được mô hình hóa bởi mất mát cross-entropy
với trọng số
quá lớn.

## Mô Hình GloVe

Từ góc nhìn này,
mô hình *GloVe* thực hiện ba thay đổi
với mô hình skip-gram dựa trên mất mát bình phương [Pennington.Socher.Manning.2014]:

1. Dùng các biến $p'_{ij}=x_{ij}$ và $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$ không phải là phân bố xác suất và lấy logarit của cả hai, nên hạng tử mất mát bình phương là $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$.
2. Thêm hai tham số mô hình vô hướng cho mỗi từ $w_i$: bias từ trung tâm $b_i$ và bias từ ngữ cảnh $c_i$.
3. Thay trọng số của mỗi hạng tử mất mát bằng hàm trọng số $h(x_{ij})$, trong đó $h(x)$ tăng trên khoảng $[0, 1]$.

Gộp tất cả lại, huấn luyện GloVe là tối thiểu hóa hàm mất mát sau:

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$

Với hàm trọng số, một lựa chọn được gợi ý là:
$h(x) = (x/c) ^\alpha$ (ví dụ $\alpha = 0.75$) nếu $x < c$ (ví dụ, $c = 100$); ngược lại $h(x) = 1$.
Trong trường hợp này,
vì $h(0)=0$,
hạng tử mất mát bình phương cho bất kỳ $x_{ij}=0$ nào có thể được bỏ qua
để tính toán hiệu quả hơn.
Ví dụ,
khi dùng minibatch stochastic gradient descent để huấn luyện,
tại mỗi vòng lặp,
chúng ta lấy mẫu ngẫu nhiên một minibatch các $x_{ij}$ *khác không*
để tính gradient
và cập nhật tham số mô hình.
Lưu ý rằng các $x_{ij}$ khác không này là các thống kê toàn cục của kho ngữ liệu đã được tính trước;
do đó, mô hình được gọi là GloVe
viết tắt của *Global Vectors*.

Cần nhấn mạnh rằng
nếu từ $w_i$ xuất hiện trong cửa sổ ngữ cảnh của
từ $w_j$, thì *ngược lại* cũng vậy.
Do đó, $x_{ij}=x_{ji}$.
Khác với word2vec,
vốn khớp xác suất có điều kiện bất đối xứng
$p_{ij}$,
GloVe khớp $\log \, x_{ij}$ đối xứng.
Vì vậy, vector từ trung tâm và
vector từ ngữ cảnh của bất kỳ từ nào là tương đương về mặt toán học trong mô hình GloVe.
Tuy nhiên trong thực tế, do các giá trị khởi tạo khác nhau,
cùng một từ vẫn có thể nhận các giá trị khác nhau
trong hai vector này sau khi huấn luyện:
GloVe cộng chúng lại làm vector đầu ra.


## Diễn Giải GloVe Từ Tỉ Số Xác Suất Đồng Xuất Hiện


Chúng ta cũng có thể diễn giải mô hình GloVe từ một góc nhìn khác.
Dùng cùng ký hiệu trong
[subsec_skipgram-global](#subsec_skipgram-global),
đặt $p_{ij} \stackrel{\textrm{def}}{=} P(w_j \mid w_i)$ là xác suất có điều kiện để sinh từ ngữ cảnh $w_j$ khi biết $w_i$ là từ trung tâm trong kho ngữ liệu.
[tab_glove](#tab_glove)
liệt kê một số xác suất đồng xuất hiện
khi biết các từ "ice" và "steam"
cùng tỉ số của chúng dựa trên thống kê từ một kho ngữ liệu lớn.


:Xác suất đồng xuất hiện giữa từ với từ và tỉ số của chúng từ một kho ngữ liệu lớn (phỏng theo Bảng 1 trong Pennington.Socher.Manning.2014)
<a id="tab_glove"></a>

|$w_k$=|solid|gas|water|fashion|
|:--|:-|:-|:-|:-|
|$p_1=P(w_k\mid \textrm{ice})$|0.00019|0.000066|0.003|0.000017|
|$p_2=P(w_k\mid\textrm{steam})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|


Chúng ta có thể quan sát các điểm sau từ [tab_glove](#tab_glove):

* Với một từ $w_k$ liên quan đến "ice" nhưng không liên quan đến "steam", chẳng hạn $w_k=\textrm{solid}$, chúng ta kỳ vọng tỉ số xác suất đồng xuất hiện lớn hơn, chẳng hạn 8.9.
* Với một từ $w_k$ liên quan đến "steam" nhưng không liên quan đến "ice", chẳng hạn $w_k=\textrm{gas}$, chúng ta kỳ vọng tỉ số xác suất đồng xuất hiện nhỏ hơn, chẳng hạn 0.085.
* Với một từ $w_k$ liên quan đến cả "ice" và "steam", chẳng hạn $w_k=\textrm{water}$, chúng ta kỳ vọng tỉ số xác suất đồng xuất hiện gần 1, chẳng hạn 1.36.
* Với một từ $w_k$ không liên quan đến cả "ice" và "steam", chẳng hạn $w_k=\textrm{fashion}$, chúng ta kỳ vọng tỉ số xác suất đồng xuất hiện gần 1, chẳng hạn 0.96.


Có thể thấy rằng tỉ số
của các xác suất đồng xuất hiện
có thể biểu đạt trực quan
mối quan hệ giữa các từ.
Do đó, chúng ta có thể thiết kế một hàm
của ba vector từ
để khớp tỉ số này.
Với tỉ số xác suất đồng xuất hiện
${p_{ij}}/{p_{ik}}$
khi $w_i$ là từ trung tâm
và $w_j$ cùng $w_k$ là các từ ngữ cảnh,
chúng ta muốn khớp tỉ số này
bằng một hàm nào đó $f$:

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$

Trong nhiều thiết kế khả dĩ cho $f$,
chúng ta chỉ chọn một lựa chọn hợp lý sau đây.
Vì tỉ số xác suất đồng xuất hiện
là một vô hướng,
chúng ta yêu cầu
$f$ là một hàm vô hướng, chẳng hạn
$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$.
Đổi chỉ số từ
$j$ và $k$ trong :eqref:`eq_glove-f`,
cần có
$f(x)f(-x)=1$,
vì vậy một khả năng là $f(x)=\exp(x)$,
tức là

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$

Bây giờ hãy chọn
$\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$,
trong đó $\alpha$ là một hằng số.
Vì $p_{ij}=x_{ij}/x_i$, sau khi lấy logarit hai vế, ta được $\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$.
Chúng ta có thể dùng các hạng tử bias bổ sung để khớp $- \log\, \alpha + \log\, x_i$, chẳng hạn bias từ trung tâm $b_i$ và bias từ ngữ cảnh $c_j$:

$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log\, x_{ij}.$$

Đo sai số bình phương của
:eqref:`eq_glove-square` với trọng số,
ta thu được hàm mất mát GloVe trong
:eqref:`eq_glove-loss`.


## Tóm Tắt

* Mô hình skip-gram có thể được diễn giải bằng các thống kê toàn cục của kho ngữ liệu như số lần đồng xuất hiện giữa từ với từ.
* Mất mát cross-entropy có thể không phải là lựa chọn tốt để đo sai khác giữa hai phân bố xác suất, đặc biệt với một kho ngữ liệu lớn. GloVe dùng mất mát bình phương để khớp các thống kê toàn cục của kho ngữ liệu đã được tính trước.
* Vector từ trung tâm và vector từ ngữ cảnh là tương đương về mặt toán học với bất kỳ từ nào trong GloVe.
* GloVe có thể được diễn giải từ tỉ số xác suất đồng xuất hiện giữa từ với từ.


## Bài Tập

1. Nếu các từ $w_i$ và $w_j$ đồng xuất hiện trong cùng một cửa sổ ngữ cảnh, làm thế nào để dùng khoảng cách của chúng trong chuỗi văn bản để thiết kế lại phương pháp tính xác suất có điều kiện $p_{ij}$? Gợi ý: xem Mục 4.2 của bài báo GloVe [Pennington.Socher.Manning.2014].
1. Với bất kỳ từ nào, bias từ trung tâm và bias từ ngữ cảnh của nó có tương đương về mặt toán học trong GloVe không? Vì sao?


[Thảo luận](https://discuss.d2l.ai/t/385)
