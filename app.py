import os
import json
import time
from flask import Flask, render_template, request, jsonify, send_file
from openai import AzureOpenAI
import tempfile
from datetime import datetime

app = Flask(__name__)

# Azure OpenAI 配置
client = AzureOpenAI(
    api_key="",
    azure_endpoint="",
    api_version="2025-01-01-preview",  # 请替换为您的API版本
)


def chat_completion(messages, model="gpt-4o", temperature=0.85, retry_count=5):
    """调用GPT进行对话分析"""
    if retry_count == 0:
        return ""
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=4000,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            messages=messages
        )
        res_content = response.choices[0].message.content.strip()
        return res_content
    except Exception as e:
        print(f"API调用错误: {e}")
        if retry_count > 1:
            time.sleep(2)  # 等待2秒后重试
            return chat_completion(messages, model, temperature, retry_count - 1)
        return ""


def load_knowledge_base():
    """加载知识库文件"""
    try:
        # 加载评估标准
        with open('data/医药销售拜访评估表.json', 'r', encoding='utf-8') as f:
            evaluation_standards = json.load(f)

        # 加载安维汀专业知识
        with open('data/安维汀.json', 'r', encoding='utf-8') as f:
            avastin_knowledge = json.load(f)

        return evaluation_standards, avastin_knowledge
    except Exception as e:
        print(f"加载知识库失败: {e}")
        return None, None


def parse_dialogue_input(dialogue_text):
    """解析用户输入的对话文本"""
    try:
        # 尝试解析为JSON格式
        dialogue_data = json.loads(dialogue_text)
        if "对话记录" in dialogue_data:
            return dialogue_data
        else:
            # 如果不是标准格式，创建一个基本结构
            return {
                "产品名称": "安维汀",
                "对话记录": dialogue_data if isinstance(dialogue_data, list) else []
            }
    except json.JSONDecodeError:
        # 如果不是JSON格式，尝试解析为纯文本
        lines = dialogue_text.strip().split('\n')
        dialogue_records = []

        for i, line in enumerate(lines):
            if line.strip():
                # 简单的文本解析逻辑
                if ':' in line or '：' in line:
                    parts = line.replace('：', ':').split(':', 1)
                    if len(parts) == 2:
                        speaker = parts[0].strip()
                        content = parts[1].strip()
                        dialogue_records.append({
                            "序号": i + 1,
                            "发言人": speaker,
                            "内容": content
                        })

        return {
            "产品名称": "安维汀",
            "对话记录": dialogue_records
        }


def stage1_basic_analysis(dialogue_data, evaluation_standards):
    """阶段一：基础拜访技巧分析"""

    # 构造对话文本
    dialogue_text = ""
    for record in dialogue_data.get("对话记录", []):
        speaker = record.get("发言人", "")
        content = record.get("内容", "")
        dialogue_text += f"{speaker}: {content}\n"

    # 获取推广目标
    promotion_goal = dialogue_data.get('推广目标', '提升产品使用')

    prompt = f"""你是一位资深的医药销售培训专家，专门负责分析和评估医药代表的拜访技巧。

请根据以下医药销售拜访评估标准，对提供的对话进行详细分析：

评估标准：
{json.dumps(evaluation_standards, ensure_ascii=False, indent=2)}

需要分析的对话：
产品名称：{dialogue_data.get('产品名称', '安维汀')}
推广目标：{promotion_goal}

对话内容：
{dialogue_text}

请按照以下结构进行分析，并以JSON格式输出：

{{
  "减分项分析": {{
    "分析项目": [
      {{
        "减分项": "具体的减分行为",
        "对话体现": "对应的对话片段",
        "原因分析": {{
          "对医生价值的损害": {{
            "损害类型": "具体损害类型",
            "具体表现": "详细说明"
          }},
          "对销售价值的损害": {{
            "损害类型": "具体损害类型", 
            "具体表现": "详细说明"
          }}
        }},
        "涉及阶段": "对应的拜访阶段"
      }}
    ]
  }},
  "加分项分析": {{
    "分析项目": [
      {{
        "加分项": "具体的加分行为",
        "对话体现": "对应的对话片段",
        "原因分析": {{
          "对医生价值": {{
            "价值类型": "具体价值类型",
            "具体表现": "详细说明"
          }},
          "对销售价值": {{
            "价值类型": "具体价值类型",
            "具体表现": "详细说明"
          }}
        }},
        "涉及阶段": "对应的拜访阶段"
      }}
    ]
  }},
  "医生的关键顾虑": {{
    "顾虑1": "提取的医生顾虑",
    "顾虑2": "提取的医生顾虑",
    "顾虑3": "提取的医生顾虑"
  }},
  "推广目标完成度评估": {{
    "推广目标": "{promotion_goal}",
    "完成度分析": "基于对话内容分析推广目标的完成情况",
    "关键成果": "达成的关键成果或承诺",
    "未达成因素": "影响目标完成的主要因素"
  }},
  "总结评估": {{
    "整体特征": "总体评价",
    "优点": ["优点1", "优点2"],
    "核心问题": ["问题1", "问题2"],
    "根本问题": "根本问题描述",
    "改进建议": "具体的改进建议"
  }}
}}

注意：
1. 请仔细分析每个拜访阶段的表现
2. 重点关注销售代表的沟通技巧、异议处理能力、产品介绍方式等
3. 分析要客观、具体，有明确的对话依据
4. 输出必须是有效的JSON格式
"""

    messages = [{"role": "user", "content": prompt}]
    response = chat_completion(messages)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # 如果返回的不是有效JSON，尝试提取JSON部分
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass

        # 如果解析失败，返回基础结构
        return {
            "减分项分析": {"分析项目": []},
            "加分项分析": {"分析项目": []},
            "医生的关键顾虑": {},
            "总结评估": {"整体特征": "分析失败", "优点": [], "核心问题": [], "根本问题": "", "改进建议": ""},
            "原始回复": response
        }


def stage2_professional_analysis(dialogue_data, basic_analysis, avastin_knowledge):
    """阶段二：专业知识评判"""

    dialogue_text = ""
    for record in dialogue_data.get("对话记录", []):
        speaker = record.get("发言人", "")
        content = record.get("内容", "")
        dialogue_text += f"{speaker}: {content}\n"

    prompt = f"""你是一位医药专业知识专家，现在需要你基于已有的基础分析，补充关于药品专业性的评判。

基础分析结果：
{json.dumps(basic_analysis, ensure_ascii=False, indent=2)}

安维汀专业知识库：
{json.dumps(avastin_knowledge, ensure_ascii=False, indent=2)}

原始对话：
{dialogue_text}

请重点分析销售代表在以下方面的专业表现，并以JSON格式输出药品专业性评判：

{{
  "药品专业性评判": {{
    "产品信息准确性": [
      {{
        "销售代表声称": "具体的产品信息声称",
        "专业评判": "准确/基本准确/不准确/过度承诺",
        "依据": "基于专业知识的评判依据"
      }}
    ],
    "医生顾虑的专业回应": [
      {{
        "医生顾虑": "医生提出的具体顾虑",
        "销售代表回应": "销售代表的回应",
        "专业评判": "回应质量评价",
        "改进建议": "基于专业知识的改进建议"
      }}
    ],
    "遗漏的专业机会": [
      {{
        "机会点": "未充分利用的专业机会",
        "专业建议": "基于知识库的建议"
      }}
    ],
    "总体专业性评估": "销售代表专业能力的总体评价"
  }}
}}

重点评判：
1. 销售代表提到的产品信息是否准确（剂量、疗效数据、安全性等）
2. 对医生顾虑的回应是否专业、有依据
3. 是否遗漏了重要的专业信息或机会点
4. 总体专业素养水平

请确保输出有效的JSON格式。
"""

    messages = [{"role": "user", "content": prompt}]
    response = chat_completion(messages)

    try:
        return json.loads(response)
    except json.JSONDecodeError:
        import re
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass

        return {
            "药品专业性评判": {
                "产品信息准确性": [],
                "医生顾虑的专业回应": [],
                "遗漏的专业机会": [],
                "总体专业性评估": "分析失败"
            },
            "原始回复": response
        }


def merge_analyses(basic_analysis, professional_analysis, dialogue_data):
    """合并两阶段的分析结果并计算推广目标得分"""
    final_analysis = {
        "标题": "医药销售拜访对话复盘分析",
        "分析时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "分析标准来源": "基于拜访评估标准体系的五个阶段：破冰与价值开场、深度需求探询、价值信息传递、专业异议处理、承诺与行动缔结"
    }

    # 合并基础分析
    final_analysis.update(basic_analysis)

    # 合并专业分析
    if "药品专业性评判" in professional_analysis:
        final_analysis.update(professional_analysis)

    # 计算推广目标得分
    score = calculate_promotion_score(basic_analysis, professional_analysis, dialogue_data)
    final_analysis["score"] = score

    return final_analysis


def calculate_promotion_score(basic_analysis, professional_analysis, dialogue_data):
    """根据推广目标完成情况计算得分 (0-100分)"""
    try:
        base_score = 60  # 基础分数

        # 加分项评估 (最多+20分)
        add_points = 0
        if basic_analysis.get("加分项分析", {}).get("分析项目"):
            add_points = min(len(basic_analysis["加分项分析"]["分析项目"]) * 5, 20)

        # 减分项扣分 (最多-30分)
        minus_points = 0
        if basic_analysis.get("减分项分析", {}).get("分析项目"):
            minus_points = min(len(basic_analysis["减分项分析"]["分析项目"]) * 8, 30)

        # 专业性加分 (最多+10分)
        professional_points = 0
        if professional_analysis.get("药品专业性评判", {}).get("总体专业性评估"):
            eval_text = professional_analysis["药品专业性评判"]["总体专业性评估"]
            if "优秀" in eval_text or "专业" in eval_text:
                professional_points = 10
            elif "良好" in eval_text:
                professional_points = 5

        # 推广目标完成度加分 (最多+20分)
        goal_points = 0
        if basic_analysis.get("推广目标完成度评估"):
            completion_text = basic_analysis["推广目标完成度评估"].get("完成度分析", "")
            if "成功" in completion_text or "达成" in completion_text:
                goal_points = 20
            elif "部分" in completion_text:
                goal_points = 10
            elif "未达成" in completion_text or "失败" in completion_text:
                goal_points = -10

        final_score = base_score + add_points - minus_points + professional_points + goal_points
        return max(0, min(100, final_score))  # 确保分数在0-100之间

    except Exception as e:
        print(f"计算得分时出错: {e}")
        return 60  # 默认分数


@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """分析对话"""
    try:
        print("=== 开始分析请求 ===")

        # 获取用户输入
        data = request.get_json()
        print(f"收到请求数据: {data}")

        dialogue_text = data.get('dialogue', '') if data else ''
        print(f"对话文本长度: {len(dialogue_text)}")

        if not dialogue_text.strip():
            print("错误: 对话内容为空")
            return jsonify({"error": "请输入对话内容"}), 400

        # 加载知识库
        print("正在加载知识库...")
        evaluation_standards, avastin_knowledge = load_knowledge_base()
        if not evaluation_standards or not avastin_knowledge:
            print("错误: 知识库加载失败")
            return jsonify({"error": "知识库加载失败，请检查data目录下的JSON文件"}), 500

        print("知识库加载成功")

        # 解析对话输入
        print("正在解析对话输入...")
        dialogue_data = parse_dialogue_input(dialogue_text)
        print(
            f"解析结果: 产品名称={dialogue_data.get('产品名称')}, 推广目标={dialogue_data.get('推广目标')}, 对话记录数量={len(dialogue_data.get('对话记录', []))}")

        # 阶段一：基础分析
        print("=== 开始阶段一：基础拜访技巧分析 ===")
        basic_analysis = stage1_basic_analysis(dialogue_data, evaluation_standards)
        print("阶段一分析完成")

        # 阶段二：专业分析
        print("=== 开始阶段二：专业知识评判 ===")
        professional_analysis = stage2_professional_analysis(dialogue_data, basic_analysis, avastin_knowledge)
        print("阶段二分析完成")

        # 合并分析结果
        print("正在合并分析结果...")
        final_analysis = merge_analyses(basic_analysis, professional_analysis, dialogue_data)
        print(f"最终分析完成，得分: {final_analysis.get('score', 'N/A')}")

        print("=== 分析请求完成 ===")
        return jsonify({
            "success": True,
            "analysis": final_analysis
        })

    except Exception as e:
        print(f"分析过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"分析失败: {str(e)}"}), 500


@app.route('/download/<analysis_id>')
def download_analysis(analysis_id):
    """下载分析结果"""
    # 这里简化处理，实际应用中可能需要存储分析结果
    # 暂时从session或临时存储中获取
    return jsonify({"error": "下载功能待实现"}), 501


if __name__ == '__main__':
    app.run(debug=True, port=5000)