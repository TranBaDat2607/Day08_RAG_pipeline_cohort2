# RAG Evaluation Results — DeepEval

- **Framework**: DeepEval (LLM-as-judge = `gpt-4o-mini`)
- **Generation model**: `gpt-4o-mini`, temperature=0.3, top_p=0.9
- **Golden dataset**: 18 cặp Q&A (legal + news corpus)
- **top_k**: 5 · **pass threshold**: 0.7

4 metric: **Faithfulness** (answer bám context?), **Answer Relevancy** (answer đúng câu hỏi?), **Context Recall** (retriever lấy đủ evidence?), **Context Precision** (% context hữu ích, đúng thứ hạng?).

## 1. Bảng điểm tổng hợp & So sánh A/B

| Metric | hybrid rerank | dense only | Δ (A−B) |
|---|---|---|---|
| Faithfulness | 0.810 (72% pass) | 0.823 (69% pass) | -0.013 |
| Answer Relevancy | 0.903 (83% pass) | 0.857 (78% pass) | +0.046 |
| Context Recall | 0.965 (94% pass) | 0.978 (94% pass) | -0.013 |
| Context Precision | 0.905 (89% pass) | 0.903 (89% pass) | +0.002 |
| **Overall avg** | **0.896** | **0.890** | **+0.006** |

> Config A = Hybrid (semantic ∥ lexical) + RRF + MMR rerank
>
> Config B = Dense-only (semantic search, không lexical, không rerank)

**Kết luận A/B:** Config **A_hybrid_rerank** (hybrid rerank) đạt overall cao hơn (0.896 vs 0.890).

## 2. Phân tích Worst Performers

### Config A_hybrid_rerank — hybrid rerank

| Câu hỏi | Avg | Faith | Rel | Recall | Prec | Vấn đề chính |
|---|---|---|---|---|---|---|
| Rapper Bình Gold bị phát hiện dương tính với loại chất … | 0.57 | 0.00 | 1.00 | 0.38 | 0.92 | The score is 0.00 because the actual output inaccurately states that Bình Gold tested positive for drugs including marijuana and bồ đà, whil… |
| Luật Phòng chống ma tuý 2021 quy định những biện pháp c… | 0.73 | 0.71 | 1.00 | 1.00 | 0.20 | The score is 0.20 because the relevant node, which is the fifth node, is significantly outnumbered by irrelevant nodes that do not address t… |
| Hình phạt cho tội tàng trữ trái phép chất ma tuý theo Đ… | 0.81 | 0.40 | 0.83 | 1.00 | 1.00 | The score is 0.40 because the actual output includes additional penalties such as fines, bans from professions, and asset confiscation, whic… |

### Config B_dense_only — dense only

| Câu hỏi | Avg | Faith | Rel | Recall | Prec | Vấn đề chính |
|---|---|---|---|---|---|---|
| Luật Phòng chống ma tuý 2021 quy định những biện pháp c… | 0.64 | 0.67 | 0.89 | 1.00 | 0.00 | The score is 0.00 because all nodes in the retrieval contexts are irrelevant to the input question about specific measures for drug rehabili… |
| Rapper Bình Gold bị phát hiện dương tính với loại chất … | 0.75 | 0.00 | 1.00 | 1.00 | 1.00 | The score is 0.00 because the actual output inaccurately states that Bình Gold tested positive for drugs including bồ đà, while the retrieva… |
| Người sử dụng trái phép chất ma tuý lần đầu nhưng chưa … | 0.81 | 0.75 | 1.00 | 1.00 | 0.50 | The score is 0.50 because while the second node provides a direct answer to the question regarding penalties for first-time drug use without… |

## 3. Đề xuất cải tiến

Xếp theo mức độ ưu tiên (metric yếu nhất trước):

1. **Faithfulness thấp** → answer có chi tiết không nằm trong context (hallucination). Siết SYSTEM_PROMPT ('chỉ dùng thông tin trong context, không suy diễn'), hạ temperature, và thêm bước hậu kiểm: yêu cầu LLM trích nguyên văn câu hỗ trợ cho mỗi citation. *(điểm hiện tại: 0.810)*
2. **Answer Relevancy thấp** → câu trả lời lan man / lạc đề. Yêu cầu trả lời trực tiếp, ngắn gọn trước khi diễn giải; cắt các đoạn không liên quan câu hỏi. *(điểm hiện tại: 0.903)*
3. **Context Precision thấp** → chunk hữu ích không được xếp lên đầu / lẫn nhiễu. Thêm cross-encoder reranker (Jina/Qwen) thay vì chỉ MMR, giảm top_k, và lọc chunk rác (mục lục, footer báo, danh sách link) ở bước chuẩn hoá. *(điểm hiện tại: 0.905)*
4. **Context Recall thấp** → retriever bỏ sót evidence. Tăng top_k / fetch_k, cải thiện chunking (tách theo Điều/Khoản để 1 chunk = 1 đơn vị pháp lý trọn vẹn), thử embedding mạnh hơn (bge-m3 / e5-large) và query expansion cho câu hỏi tiếng Việt. *(điểm hiện tại: 0.965)*

**Lựa chọn cấu hình:** dùng **A_hybrid_rerank** (hybrid rerank) làm mặc định production vì overall cao nhất (0.896). Các cấu hình còn lại có thể dùng làm fallback hoặc cho câu hỏi đặc thù.

**Mở rộng dataset:** golden set hiện thiên về câu hỏi factual đơn lẻ; nên bổ sung câu hỏi multi-hop (so sánh nhiều Điều luật), câu hỏi không có trong corpus (kiểm tra khả năng từ chối 'không xác minh được'), và câu hỏi follow-up để đánh giá conversation memory.

## 4. Chi tiết từng câu hỏi (Config A_hybrid_rerank)

**Q1. Hình phạt cho tội tàng trữ trái phép chất ma tuý theo Điều 249 Bộ luật Hình sự là gì?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=0.400, Answer Relevancy=0.833, Context Recall=1.000, Context Precision=1.000
- *Answer*: Theo Điều 249 Bộ luật Hình sự năm 2015 (sửa đổi, bổ sung 2025), hình phạt cho tội tàng trữ trái phép chất ma túy được quy định như sau: người nào tàng trữ trái phép chất ma túy mà không nhằm mục đích mua bán, vận chuyển, sản xuất trái phép chất ma túy sẽ bị phạt tù từ 3 năm đến 5 năm [Document 5].  …

**Q2. Tội mua bán trái phép chất ma tuý theo Điều 251 bị xử lý như thế nào?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=0.800, Answer Relevancy=0.833, Context Recall=1.000, Context Precision=1.000
- *Answer*: Tội mua bán trái phép chất ma túy được quy định tại Điều 251 của Bộ luật Hình sự 2015, sửa đổi bởi khoản 18 Điều 1 Luật sửa đổi Bộ luật Hình sự 2025. Theo quy định này, người phạm tội có thể bị phạt tù từ 3 năm đến 7 năm ở khung cơ bản [Document 1 | Source: article_03.md].   Ngoài ra, tội mua bán tr…

**Q3. Mức hình phạt cho tội tổ chức sử dụng trái phép chất ma tuý theo Điều 255 là bao nhiêu?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=0.667, Answer Relevancy=1.000, Context Recall=1.000, Context Precision=1.000
- *Answer*: Mức hình phạt cho tội tổ chức sử dụng trái phép chất ma túy theo Điều 255 Bộ luật Hình sự 2015 sửa đổi bởi khoản 19 Điều 1 Luật sửa đổi Bộ luật Hình sự 2025 là từ 3 năm tù giam đến 20 năm tù giam hoặc tù chung thân. Ngoài ra, người phạm tội còn có thể bị phạt tiền từ 50 triệu đồng đến 500 triệu đồng…

**Q4. Người sử dụng trái phép chất ma tuý lần đầu nhưng chưa đến mức truy cứu hình sự bị xử phạt ra sao?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=1.000, Answer Relevancy=1.000, Context Recall=1.000, Context Precision=0.500
- *Answer*: Người sử dụng trái phép chất ma túy lần đầu nhưng chưa đến mức bị truy cứu trách nhiệm hình sự sẽ bị xử phạt theo quy định tại khoản 1 và điểm a khoản 8 Điều 30 Nghị định 282/2025/NĐ-CP. Cụ thể, người vi phạm có thể bị phạt cảnh cáo hoặc phạt tiền từ 1 triệu đồng đến 2 triệu đồng. Ngoài ra, người vi…

**Q5. Đối với heroin, cocaine hay methamphetamine, khối lượng tối thiểu để bị truy cứu trách nhiệm hình sự là bao nhiêu?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=1.000, Answer Relevancy=1.000, Context Recall=1.000, Context Precision=1.000
- *Answer*: Đối với heroin, cocaine hoặc methamphetamine, khối lượng tối thiểu để bị truy cứu trách nhiệm hình sự là từ 0,1 gam trở lên [Document 2].

**Q6. Luật Phòng chống ma tuý 2021 quy định những biện pháp cai nghiện nào?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=0.714, Answer Relevancy=1.000, Context Recall=1.000, Context Precision=0.200
- *Answer*: Luật Phòng chống ma túy 2021 quy định một số biện pháp cai nghiện ma túy nhằm đảm bảo an toàn và hiệu quả trong quá trình cai nghiện. Các biện pháp này bao gồm:  1. **Trách nhiệm của cơ sở cai nghiện**: Người đứng đầu cơ sở cai nghiện ma túy có trách nhiệm bảo đảm điều kiện hoạt động của cơ sở, niêm…

**Q7. Thời hạn cai nghiện ma tuý tự nguyện tại gia đình, cộng đồng là bao lâu?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=1.000, Answer Relevancy=0.500, Context Recall=1.000, Context Precision=1.000
- *Answer*: Thời hạn cai nghiện ma túy tự nguyện tại gia đình, cộng đồng là từ đủ 06 tháng đến 12 tháng [Luật Phòng chống ma túy 2021, Điều 30].

**Q8. Theo Luật Phòng chống ma tuý 2021, tiền chất là gì?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=1.000, Answer Relevancy=1.000, Context Recall=1.000, Context Precision=1.000
- *Answer*: Theo Luật Phòng chống ma túy 2021, tiền chất được định nghĩa là hóa chất không thể thiếu được trong quá trình điều chế, sản xuất chất ma túy và được quy định trong danh mục tiền chất do Chính phủ ban hành [Luật Phòng chống ma túy 2021, Điều 2].

**Q9. Người sử dụng trái phép chất ma tuý được định nghĩa thế nào theo Luật Phòng chống ma tuý 2021?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=0.500, Answer Relevancy=1.000, Context Recall=1.000, Context Precision=1.000
- *Answer*: Theo Luật Phòng, chống ma túy 2021, người sử dụng trái phép chất ma túy được định nghĩa là người có hành vi sử dụng chất ma túy mà không được sự cho phép của người hoặc cơ quan chuyên môn có thẩm quyền và có kết quả xét nghiệm chất ma túy trong cơ thể dương tính [Luật Phòng chống ma túy 2021, Điều 1…

**Q10. Thời hạn quản lý người sử dụng trái phép chất ma tuý là bao lâu?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=0.667, Answer Relevancy=1.000, Context Recall=1.000, Context Precision=1.000
- *Answer*: Thời hạn quản lý người sử dụng trái phép chất ma túy là 01 năm kể từ ngày Chủ tịch Ủy ban nhân dân cấp xã ra quyết định quản lý [Luật Phòng chống ma tuý 2021, Điều 23].

**Q11. Việc xét nghiệm chất ma tuý trong cơ thể được thực hiện qua những mẫu nào?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=0.833, Answer Relevancy=1.000, Context Recall=1.000, Context Precision=1.000
- *Answer*: Việc xét nghiệm chất ma túy trong cơ thể được thực hiện thông qua các mẫu vật sau:  1. Mẫu nước tiểu: Đây là phương pháp phổ biến nhất hiện nay nhờ tính tiện lợi, cho kết quả nhanh chóng và độ chính xác cao khi cần sàng lọc người sử dụng ma túy đá [Document 5]. 2. Mẫu máu: Xét nghiệm máu cũng là một…

**Q12. Ca sĩ Miu Lê bị phát hiện dương tính với những loại ma tuý nào?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=1.000, Answer Relevancy=0.750, Context Recall=1.000, Context Precision=0.750
- *Answer*: Ca sĩ Miu Lê đã bị phát hiện dương tính với ba loại ma túy, bao gồm Methamphetamine (ma túy đá), Ketamine (“ke”) và MDMA (thuốc lắc) [Document 1].

**Q13. Vụ án ma tuý liên quan ca sĩ Long Nhật và Sơn Ngọc Minh tại TPHCM có quy mô như thế nào?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=1.000, Answer Relevancy=1.000, Context Recall=1.000, Context Precision=0.917
- *Answer*: Vụ án ma túy liên quan đến ca sĩ Long Nhật và Sơn Ngọc Minh tại TPHCM có quy mô lớn. Cụ thể, Công an TP.HCM đã triệt phá một đường dây tội phạm ma túy quy mô lớn, bắt giữ tổng cộng 74 đối tượng. Trong số này, có 71 bị can đã bị khởi tố và bắt tạm giam để điều tra về các hành vi như "Mua bán trái phé…

**Q14. Ca sĩ Châu Việt Cường bị tuyên án bao nhiêu năm tù và về tội gì?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=1.000, Answer Relevancy=1.000, Context Recall=1.000, Context Precision=1.000
- *Answer*: Ca sĩ Châu Việt Cường bị Tòa án nhân dân TP Hà Nội tuyên phạt 13 năm tù về các tội “Giết người” và “Tàng trữ trái phép chất ma túy” [Document 1].

**Q15. Diễn viên hài Hữu Tín bị tuyên án bao nhiêu năm tù?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=1.000, Answer Relevancy=0.667, Context Recall=1.000, Context Precision=1.000
- *Answer*: Diễn viên hài Hữu Tín bị tuyên án 7 năm 6 tháng tù về tội "Tổ chức sử dụng trái phép chất ma túy" vào ngày 28/4/2023 [Document 2 | Source: article_04.md].

**Q16. Diễn viên Lệ Hằng bị khởi tố về tội gì và bị phát hiện bao nhiêu ma tuý?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=1.000, Answer Relevancy=0.667, Context Recall=1.000, Context Precision=1.000
- *Answer*: Diễn viên Lệ Hằng bị khởi tố về tội “Mua bán trái phép chất ma túy” và bị phát hiện có trong người 0,696 gram ma túy tổng hợp khi kiểm tra hành chính trên phố Khâm Thiên, Hà Nội [Document 2 | Source: article_01.md].

**Q17. Ma tuý đá có thời gian bán thải trong cơ thể là bao lâu và xét nghiệm nào phát hiện được lâu nhất?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=1.000, Answer Relevancy=1.000, Context Recall=1.000, Context Precision=1.000
- *Answer*: Ma túy đá có thời gian bán thải trong cơ thể thường rơi vào khoảng 10 đến 12 tiếng. Điều này có nghĩa là sau nửa ngày, lượng độc chất trong cơ thể chỉ giảm đi 50% [Document 1].   Đối với các phương pháp xét nghiệm để phát hiện ma túy đá, xét nghiệm tóc là phương pháp có khả năng phát hiện lâu nhất. …

**Q18. Rapper Bình Gold bị phát hiện dương tính với loại chất cấm nào?**

- *Retrieval source*: `hybrid` · *chunks*: 5
- *Scores*: Faithfulness=0.000, Answer Relevancy=1.000, Context Recall=0.375, Context Precision=0.917
- *Answer*: Rapper Bình Gold bị phát hiện dương tính với chất ma túy gồm cần sa và bồ đà [article_04.md].
