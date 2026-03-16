1. disaster_look_all_pic.py：通过基础提示词对 albef_feature_extractor_base、albef_vqa_vqav2、clip_ViT_B_16、clip_ViT_B_32、clip_ViT_L_14、blip_vqa_aokvqa、blip_vqa_okvqa、blip_vqa_vqav2、vilt_b32_finetuned_vqa 共 9 种视觉语言大模型在天气灾害识别任务中的性能进行测试。结果存放于程序同目录下的 result 文件夹下。

2. disaster_look_all_pic_deepseek.py：通过基础提示词对 deepseek-vl-7b-chat 大模型在天气灾害识别任务中的性能进行测试。结果存放于同目录下的 result 文件夹下，模型输出的结果文件为 deepseek_vl.txt。考虑到模型输出的结果文件中包含少量分析性内容，通过关键词提取的方法进行手工修正，保存为 result 文件夹下的 deepseek_vl_change.txt。

3. Sensitivity test/sentence structure/disaster_look_all_pic.py 和 Sensitivity test/sentence structure/disaster_look_all_pic_deepseek.py：用于测试基础提示词结构变化后 10 种视觉语言大模型在天气灾害识别任务中的性能。结果存放于程序同目录下的 result 文件夹，并将 deepseek_vl.txt 修正后的结果命名为 deepseek_vl_change.txt。

4. Sensitivity test/synonym/disaster_look_all_pic.py 和 Sensitivity test/synonym/disaster_look_all_pic_deepseek.py：用于测试基础提示词同义词变化后 10 种视觉语言大模型在天气灾害识别任务中的性能。结果存放于程序同目录下的 result 文件夹，并将 deepseek_vl.txt 修正后的结果命名为 deepseek_vl_change.txt。

5. ensemble learning/model_ten_to_one.py：将 10 种模型在基础、同义词、结构变化三种提示词中最优结果进行预测归类与独热编码合并。各模型最优结果存放于 ensemble learning/model_result 中，生成的独热编码结果存放于 ensemble learning/model_data/model10_v2.0_encoding.csv。

6. ensemble learning/cut_data.py：将 ensemble learning/model_data 中的 model10_v2.0_encoding.csv 切割为 test_set_v4.0.csv 和 train_set_v4.0.csv，并存放于相同目录下。

7. ensemble learning/onehotEncoding_blending_prompt.py：采用 blending 集成学习方法对 train_set_v4.0.csv 进行训练，并用 test_set_v4.0.csv 进行测试。权重模型和结果文件存放于 ensemble learning/best_blending 文件夹中。