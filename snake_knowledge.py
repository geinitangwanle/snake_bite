#!/usr/bin/env python3
"""
蛇类知识库数据
包含各种蛇类的详细信息、特征、栖息地和安全建议
"""

class SnakeKnowledge:
    """蛇类知识库管理类"""
    
    def __init__(self):
        self.knowledge_base = {
            '北方水蛇': {
                'description': '北方水蛇是一种常见的无毒蛇类，主要栖息在水域附近。身体适应水生环境，是优秀的游泳者。',
                'characteristics': '身体较粗壮，体长通常在60-130厘米之间。体色多变，常见棕色、灰色或橄榄色，背部有深色横纹或斑块。腹部通常为黄色或红色，有黑色斑块。',
                'habitat': '广泛分布于北美东部，喜欢生活在池塘、河流、湖泊、沼泽等水域周围。也会在水边的岩石或倒木上晒太阳。',
                'safety': '✅ 无毒蛇类，对人类无害。虽然被咬后不会中毒，但仍需清洁伤口预防感染。性情相对温和，但受惊时会释放麝香味。',
                'behavior': '主要在白天活动，是半水生蛇类。善于游泳和潜水，主要以鱼类、蛙类、蝌蚪和小型哺乳动物为食。冬季会冬眠。',
                'conservation': '种群数量稳定，无特殊保护需求。',
                'tips': '遇到时保持距离观察即可，不要突然移动惊吓它。'
            },
            
            '普通袜带蛇': {
                'description': '普通袜带蛇是北美最常见的蛇类之一，因其身体上的条纹像袜带而得名。性情温和，适应性极强。',
                'characteristics': '身体细长，体长通常在45-65厘米之间。最显著特征是身体上有三条纵向条纹：一条位于背部中央，两条位于两侧。条纹颜色通常为黄色、绿色或蓝色。',
                'habitat': '适应性极强，几乎可以在任何环境中生存：草地、花园、森林边缘、公园、农田等。从海平面到山区都有分布。',
                'safety': '✅ 无毒蛇类，性情非常温和，很少主动攻击人类。即使被咬也只需简单处理伤口。是对人类最友好的蛇类之一。',
                'behavior': '主要在白天活动，行动敏捷。主要以蚯蚓、蛞蝓、小鱼、蛙类等为食。怀孕期约2-3个月，直接产仔而非产卵。',
                'conservation': '种群数量非常稳定，分布广泛。',
                'tips': '是花园的好朋友，能帮助控制害虫。如果在花园中发现，建议让其自然离开。'
            },
            
            '德凯棕蛇': {
                'description': '德凯棕蛇是一种小型无毒蛇，因其棕色外观和德凯这位博物学家而得名。是城市环境中常见的蛇类。',
                'characteristics': '体型较小，通常体长只有20-35厘米，是北美最小的蛇类之一。体色为棕色、灰棕色或红棕色，背部中央有浅色条纹，腹部为粉色或黄色。',
                'habitat': '喜欢隐藏在落叶堆、石头下、花园覆盖物中或建筑物附近。在城市和郊区环境中很常见。',
                'safety': '✅ 完全无毒，体型极小，对人类无任何威胁。性情温和，很少咬人，即使咬了也感觉不到。',
                'behavior': '主要在夜间和黄昏活动，白天通常隐藏起来。行动相对缓慢，主要以蚯蚓、蛞蝓、小昆虫和其幼虫为食。',
                'conservation': '种群稳定，但栖息地丧失是主要威胁。',
                'tips': '是花园生态系统的重要组成部分，能有效控制土壤害虫。发现时请不要伤害。'
            },
            
            '黑鼠蛇': {
                'description': '黑鼠蛇是一种大型无毒蛇类，是优秀的攀爬者和鼠类控制专家。成年后全身呈现亮黑色。',
                'characteristics': '大型蛇类，成年体长可达1.2-2.5米。成年蛇通体黑色且有光泽，幼蛇为灰色并有深色斑纹。腹部为灰白色或黄色。身体强壮，鳞片光滑。',
                'habitat': '栖息在森林、农田、谷仓、废弃建筑等地方。善于攀爬，经常在树上或建筑物的阁楼中发现。分布范围广泛。',
                'safety': '✅ 无毒蛇类，但体型较大且力量强。受惊或感受到威胁时可能会咬人，伤口较深但无毒。建议保持安全距离。',
                'behavior': '主要在白天活动，是出色的攀爬者。主要以啮齿动物、鸟类和鸟蛋为食，是控制鼠类的天然专家。性情相对温和但会自卫。',
                'conservation': '种群相对稳定，是生态系统中重要的掠食者。',
                'tips': '对农民和房主有益，能有效控制鼠类。如在建筑物中发现，建议请专业人员安全转移。'
            },
            
            '西部菱斑响尾蛇': {
                'description': '西部菱斑响尾蛇是一种剧毒蛇类，具有标志性的响尾警告系统。是北美西部最危险的蛇类之一。',
                'characteristics': '体型粗壮，体长通常在90-150厘米之间。背部有明显的菱形斑纹，颜色从灰色到棕色不等。最显著特征是尾部的角质响环，受威胁时会快速摆动发出响声。',
                'habitat': '主要栖息在干旱和半干旱地区：沙漠、草原、岩石地带、灌木丛等。喜欢在岩石缝隙、洞穴中栖息。',
                'safety': '⚠️⚠️ 剧毒蛇类！毒液含有血循毒和神经毒素，可致命！遇到立即远离，如被咬立即拨打急救电话并前往最近的医院！',
                'behavior': '主要在夜间和黄昏活动，白天在阴凉处休息。受威胁时会抬起前身做出攻击姿态，同时快速摆动尾部发出警告。攻击距离可达体长的2/3。',
                'conservation': '种群数量因栖息地丧失而下降，但仍然常见。',
                'tips': '🚨 紧急处理：保持冷静，立即远离，不要试图捕捉或杀死。被咬后不要奔跑，立即就医！预防：在其栖息地行走时穿厚靴子，使用手电筒。'
            }
        }
    
    def get_snake_info(self, snake_name):
        """获取指定蛇类的详细信息"""
        return self.knowledge_base.get(snake_name, None)
    
    def get_all_snakes(self):
        """获取所有蛇类名称列表"""
        return list(self.knowledge_base.keys())
    
    def format_snake_info(self, snake_name):
        """格式化蛇类信息为显示文本"""
        info = self.get_snake_info(snake_name)
        if not info:
            return f"暂无 {snake_name} 的详细信息。\n\n如果您有相关资料，欢迎提供补充！"
        
        # 根据安全级别选择合适的表情符号
        safety_level = "🚨" if "剧毒" in info['safety'] else "✅"
        
        formatted_text = f"""
{safety_level} **{snake_name}** 详细档案

📖 物种介绍
{info['description']}

🎯 外观特征 
{info['characteristics']}

🏠 栖息环境
{info['habitat']}

⚠️ 安全评估
{info['safety']}

🦎 生活习性
{info['behavior']}

🌍 保护状况
{info['conservation']}

💡 实用建议
{info['tips']}
        """
        return formatted_text.strip()
    
    def search_by_keyword(self, keyword):
        """根据关键词搜索相关蛇类"""
        results = []
        keyword = keyword.lower()
        
        for snake_name, info in self.knowledge_base.items():
            # 在所有信息字段中搜索关键词
            all_text = " ".join(info.values()).lower()
            if keyword in all_text or keyword in snake_name.lower():
                results.append(snake_name)
        
        return results

# 创建全局知识库实例
snake_kb = SnakeKnowledge()

# 为了向后兼容，提供一个简单的函数接口
def get_snake_knowledge(snake_name):
    """获取蛇类知识信息 - 简化接口"""
    return snake_kb.format_snake_info(snake_name)

# 如果直接运行此文件，展示所有蛇类信息
if __name__ == "__main__":
    print("🐍 蛇类知识库测试")
    print("=" * 50)
    
    for snake_name in snake_kb.get_all_snakes():
        print(f"\n{snake_name}:")
        print("-" * 30)
        print(snake_kb.format_snake_info(snake_name))
        print("\n" + "="*50)