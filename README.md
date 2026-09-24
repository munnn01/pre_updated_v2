# pre_updated_v2 — nén hướng nhiệm vụ cho Action Recognition và Object Detection

Repo này nghiên cứu can thiệp miền pixel **trước codec chuẩn** để giảm bitrate mà vẫn giữ hiệu năng tác vụ. Hai đường đánh giá độc lập:

- **AR / Kinetics:** chọn một trong sáu biểu diễn clip bằng policy V2-C đã khóa; mã hóa H.264 hoặc H.265, giải mã, rồi đo Top-1 bằng hai mạng `r2plus1d_18` và `r3d_18` đóng băng. Kinetics và họ mạng video này được mô tả trong [1, 3]; phiên bản trọng số dùng qua TorchVision xem [4].
- **OD / COCO:** detector phía encoder tạo vùng cần bảo vệ; làm mờ nền ngoài vùng đó trước codec, rồi dùng một detector khác phía decoder để đo COCO mAP. COCO và Faster R-CNN xem [2, 5]. Đây **không** phải V2-C áp dụng sang ảnh.

## Kết quả AR được giữ trong `results/`

[Gói V2-C 1.000 clip](results/dual_codec_search_v2_confirm_1000/README.md) là gói kết quả chính duy nhất trong thư mục `results/`. Cùng 1.000 clip TEST được ghép cặp giữa hai codec, năm QP `30,35,40,45,50`, và 2.000 lần bootstrap theo **video nguồn**; BD-rate được tính sau khi gộp hai shard 500 clip, không lấy trung bình BD-rate của shard.

BD-rate ở đây áp dụng phép so sánh đường rate–quality kiểu Bjøntegaard [6] với **Top-1** (hoặc mAP ở pilot OD) làm quality; tài liệu gốc dùng PSNR. Hai codec tương ứng chuẩn ITU-T H.264 và H.265 [7, 8].

| Codec | Analyzer | BD-rate Top-1 | Bootstrap 95% | BD-accuracy (điểm %) |
|---|---|---:|---:|---:|
| H.264 | `r2plus1d_18` | **−22,01%** | [−23,85%, −20,24%] | +10,40 |
| H.264 | `r3d_18` | −14,47% | [−15,69%, −13,21%] | +6,09 |
| H.265 | `r2plus1d_18` | −14,03% | [−15,39%, −12,77%] | +9,92 |
| H.265 | `r3d_18` | −8,72% | [−9,62%, −7,82%] | +5,53 |

**Quyết định theo tiêu chí đặt trước:** chưa đạt yêu cầu cả hai analyzer đều có BD-rate Top-1 **< −15%** ở ít nhất một codec. H.264 gần nhất nhưng `r3d_18` còn thiếu 0,53 điểm phần trăm. Cả hai analyzer đều tham gia phát triển policy; hơn nữa TEST này đã được xem trong nghiên cứu V1. Đây là phép so sánh ghép cặp trên tập đã biết, **không phải** kiểm chứng độc lập trên holdout mới.

Các JSON tổng hợp [H.264](results/dual_codec_search_v2_confirm_1000/h264_result.json) và [H.265](results/dual_codec_search_v2_confirm_1000/h265_result.json) giữ curve, fingerprint và control đồng thời để kiểm toán. Gói kết quả V1 riêng đã được bỏ khỏi nhánh hiện tại; mã và cấu hình V1 cần cho mẫu ghép cặp vẫn được giữ. Những lần Kaggle lỗi không được đưa vào `results/`.

## Kiểm tra ảnh ghép cặp và OD

[Notebook Kinetics cuối cùng](https://www.kaggle.com/code/qktttttttttt/paper-ar-visual-20260924) đã hoàn tất trên CPU: tám clip được chọn bằng hash ID trước khi xem nhãn/kết quả, cùng tám ID cho hai codec, cùng QP 40 và các frame 4/8/12. Panel gồm nguồn, codec-only và stream V2-C; bpp mã hóa lại khớp chính xác cache gốc. Ảnh được phóng bằng nearest-neighbor để hiển thị, không làm đổi pixel mã hóa. **Tám clip chỉ để minh họa**, không thay phép đo Top-1 trên 1.000 clip.

[Notebook COCO val2017](https://www.kaggle.com/code/baoancut/paper-coco-visual-20260924) cũng đã hoàn tất. Đây là pilot OD **100 ảnh** tại 320 px, QP `35,40,45`, `blur4` ngoài vùng bảo vệ so với codec-only. Detector tạo mask là Faster R-CNN MobileNet; detector đánh giá độc lập là Faster R-CNN ResNet50. Tám panel ảnh được chọn bằng ID trong tập đã định, có mask và bản đồ sai khác RGB dùng chung thang 0–64.

| Codec OD | mAP tại QP 40: codec-only → `blur4` | BD-rate theo mAP, ba QP | Bootstrap 95%, 100 lần lấy mẫu ảnh |
|---|---:|---:|---:|
| H.264 | 0,1958 → 0,1886 | **−13,55%** | [−22,20%, −3,00%] |
| H.265 | 0,2126 → 0,1973 | −7,81% | [−16,09%, +2,66%] |

Ở cùng QP, mAP giảm nhẹ; BD-rate âm phản ánh tiết kiệm bit theo toàn đường rate–mAP. H.265 còn bất định vì khoảng bootstrap cắt 0. Pilot 100 ảnh này không phải xác nhận trên toàn COCO val2017 và không chứng minh một phương pháp chung cải thiện cả OD lẫn AR. Artifact đầy đủ nằm ở output Kaggle, **không** được đưa vào `results/` như kết quả chính.

## Mã và tái lập

- [Thiết kế và giới hạn nghiên cứu](docs/PAPER_VALIDATION_PLAN.md): phép so sánh cố định, bootstrap ghép cặp, phép thử analyzer thứ ba và chi phí chạy còn phải đo.
- [Policy và runner V2](ops/dual_codec_search_confirm_1000.py), [tạo panel Kinetics](ops/paper_ar_visual.py), [OD pilot và panel COCO](ops/probe_background_suppression.py).
- [Phân tích ablation](ops/paper_validation.py), [runner `mc3_18` chưa đo](ops/paper_heldout_mc3.py), [runner thời gian chạy chưa đo](ops/paper_runtime.py). Có mã không đồng nghĩa đã có kết quả thực nghiệm.
- [Cell Kaggle AR](kaggle/paper_ar_visual_cell.sh), [cell Kaggle OD](kaggle/paper_coco_visual_cell.sh) và [công cụ tạo notebook riêng tư](ops/push_paper_visual.py). Cell trong repo này clone `munnn01/pre_updated_v2` tại commit được chỉ định. Hai notebook hoàn tất ở trên được chạy từ bản phát triển `test_pre` với cùng logic đánh giá; không được gọi là lượt chạy lại trên commit repo này.

Các bước cần làm trước khi tuyên bố khả năng tổng quát: đánh giá bitstream đã chọn bằng analyzer thứ ba chưa tham gia phát triển, thử trên nguồn video mới tách hẳn, và đo chi phí của **toàn bộ** sáu phép encode/decode cùng suy luận tại encoder. Chưa có số liệu cho ba bước đó.

## Tài liệu tham khảo

Các nguồn dưới đây là nền tảng cho dữ liệu, mô hình, codec và thước đo; **không phải nguồn của các con số thực nghiệm** trong bảng. Số liệu của repo được lưu trong JSON và notebook liên kết ở trên.

1. Kay, W. và cộng sự (2017). *The Kinetics Human Action Video Dataset*. [arXiv:1705.06950](https://arxiv.org/abs/1705.06950). Bản dữ liệu thực nghiệm: [Kinetics cleaned trên Kaggle](https://www.kaggle.com/datasets/qktttttttttt/kineticscleaned); không đồng nhất bản cleaned này với tập gốc trong bài báo.
2. Lin, T.-Y. và cộng sự (2014). *Microsoft COCO: Common Objects in Context*. [arXiv:1405.0312](https://arxiv.org/abs/1405.0312). Bản dữ liệu thực nghiệm: [COCO 2017 trên Kaggle](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset).
3. Tran, D. và cộng sự (2018). *A Closer Look at Spatiotemporal Convolutions for Action Recognition*. CVPR, trang 6450–6459. [CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html).
4. TorchVision. Tài liệu và trọng số tiền huấn luyện Kinetics-400 cho [`r2plus1d_18`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r2plus1d_18.html) và [`r3d_18`](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.video.r3d_18.html).
5. Ren, S., He, K., Girshick, R. và Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. NeurIPS 28. [Bài báo](https://papers.nips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html); triển khai TorchVision: [MobileNetV3-FPN](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn.html), [ResNet50-FPN](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html).
6. Bjøntegaard, G. (2001). *Calculation of Average PSNR Differences between RD-curves*. ITU-T VCEG-M33. [Bản tài liệu gốc](https://eclass.uoa.gr/modules/document/file.php/D221/%CE%A3%CE%B7%CE%BC%CE%B5%CE%B9%CF%8E%CF%83%CE%B5%CE%B9%CF%82/VCEG-M33%20%28Bjontegaard%20Delta%29.pdf).
7. ITU-T. *Recommendation H.264: Advanced video coding for generic audiovisual services*. [Trang chuẩn](https://www.itu.int/rec/t-rec-h.264).
8. ITU-T. *Recommendation H.265: High efficiency video coding*. [Trang chuẩn](https://www.itu.int/rec/T-REC-H.265).
