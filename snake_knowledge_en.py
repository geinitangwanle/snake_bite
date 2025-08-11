#!/usr/bin/env python3
"""
Snake Knowledge Database
Contains detailed information about various snake species including characteristics, habitats, and safety advice
"""

class SnakeKnowledge:
    """Snake Knowledge Database Management Class"""
    
    def __init__(self):
        self.knowledge_base = {
            'Northern Water Snake': {
                'description': 'The Northern Water Snake is a common non-venomous snake species that primarily inhabits areas near water bodies. Its body is well-adapted to aquatic environments and is an excellent swimmer.',
                'characteristics': 'Has a relatively robust body, typically 60-130 cm in length. Body color varies, commonly brown, gray, or olive, with dark crossbands or patches on the back. The belly is usually yellow or red with black patches.',
                'habitat': 'Widely distributed in eastern North America, preferring to live around ponds, rivers, lakes, swamps, and other water bodies. Also basks on rocks or fallen logs near water.',
                'safety': '✅ Non-venomous snake, harmless to humans. Although bites won\'t cause poisoning, wounds should still be cleaned to prevent infection. Relatively mild-tempered but releases musky odor when frightened.',
                'behavior': 'Primarily active during the day, semi-aquatic snake. Excellent at swimming and diving, mainly feeds on fish, frogs, tadpoles, and small mammals. Hibernates in winter.',
                'conservation': 'Population is stable with no special conservation needs.',
                'tips': 'Maintain distance and observe when encountered, avoid sudden movements that might startle it.'
            },
            
            'Common Garter Snake': {
                'description': 'The Common Garter Snake is one of the most common snake species in North America, named for the stripes on its body that resemble garters. Very mild-tempered with excellent adaptability.',
                'characteristics': 'Slender body, typically 45-65 cm in length. Most distinctive feature is three longitudinal stripes: one along the center of the back and two on the sides. Stripe colors are usually yellow, green, or blue.',
                'habitat': 'Extremely adaptable, can survive in almost any environment: grasslands, gardens, forest edges, parks, farmlands, etc. Distributed from sea level to mountainous regions.',
                'safety': '✅ Non-venomous snake with very mild temperament, rarely attacks humans actively. Even if bitten, only simple wound care is needed. One of the most human-friendly snake species.',
                'behavior': 'Primarily active during the day, moves agilely. Mainly feeds on earthworms, slugs, small fish, frogs, etc. Gestation period is about 2-3 months, gives birth to live young rather than laying eggs.',
                'conservation': 'Population is very stable with wide distribution.',
                'tips': 'A garden\'s best friend, helps control pests. If found in gardens, recommend letting it leave naturally.'
            },
            
            'DeKay\'s Brownsnake': {
                'description': 'DeKay\'s Brownsnake is a small non-venomous snake, named for its brown appearance and the naturalist DeKay. Common snake species in urban environments.',
                'characteristics': 'Small size, usually only 20-35 cm in length, one of the smallest snake species in North America. Body color is brown, gray-brown, or reddish-brown with a light stripe down the center of the back, belly is pink or yellow.',
                'habitat': 'Likes to hide under leaf litter, stones, garden mulch, or near buildings. Very common in urban and suburban environments.',
                'safety': '✅ Completely non-venomous, extremely small size poses no threat to humans. Mild-tempered, rarely bites, and even if it does, it\'s barely noticeable.',
                'behavior': 'Primarily active at night and dusk, usually hidden during the day. Moves relatively slowly, mainly feeds on earthworms, slugs, small insects and their larvae.',
                'conservation': 'Population is stable, but habitat loss is the main threat.',
                'tips': 'Important component of garden ecosystem, effectively controls soil pests. Please don\'t harm when discovered.'
            },
            
            'Black Rat Snake': {
                'description': 'The Black Rat Snake is a large non-venomous snake species, an excellent climber and rodent control expert. Adults display a bright black coloration throughout their body.',
                'characteristics': 'Large snake species, adult body length can reach 1.2-2.5 meters. Adult snakes are entirely black and glossy, juveniles are gray with dark markings. Belly is grayish-white or yellow. Strong body with smooth scales.',
                'habitat': 'Inhabits forests, farmlands, barns, abandoned buildings, etc. Excellent climber, often found in trees or building attics. Wide distribution range.',
                'safety': '✅ Non-venomous snake, but large size and strong. May bite when startled or threatened, wounds are deep but non-toxic. Recommend maintaining safe distance.',
                'behavior': 'Primarily active during the day, excellent climber. Mainly feeds on rodents, birds, and bird eggs, natural expert at rodent control. Relatively mild-tempered but will defend itself.',
                'conservation': 'Population is relatively stable, important predator in ecosystem.',
                'tips': 'Beneficial to farmers and homeowners, effectively controls rodents. If found in buildings, recommend professional safe relocation.'
            },
            
            'Western Diamondback Rattlesnake': {
                'description': 'The Western Diamondback Rattlesnake is a highly venomous snake species with a characteristic rattle warning system. One of the most dangerous snakes in western North America.',
                'characteristics': 'Robust build, typically 90-150 cm in length. Distinctive diamond-shaped patterns on the back, colors ranging from gray to brown. Most distinctive feature is the horny rattle segments at the tail tip, which vibrate rapidly when threatened to produce warning sounds.',
                'habitat': 'Primarily inhabits arid and semi-arid regions: deserts, grasslands, rocky areas, scrublands, etc. Prefers to shelter in rock crevices and caves.',
                'safety': '⚠️⚠️ HIGHLY VENOMOUS SNAKE! Venom contains hemotoxins and neurotoxins, potentially fatal! Immediately retreat when encountered, if bitten immediately call emergency services and go to nearest hospital!',
                'behavior': 'Primarily active at night and dusk, rests in shaded areas during the day. When threatened, raises front body in striking position while rapidly vibrating tail as warning. Strike distance can reach 2/3 of body length.',
                'conservation': 'Population declining due to habitat loss, but still common.',
                'tips': '🚨 Emergency Treatment: Stay calm, immediately retreat, do not attempt to capture or kill. If bitten, do not run, seek immediate medical attention! Prevention: Wear thick boots when walking in their habitat, use flashlights.'
            }
        }
    
    def get_snake_info(self, snake_name):
        """Get detailed information for specified snake species"""
        return self.knowledge_base.get(snake_name, None)
    
    def get_all_snakes(self):
        """Get list of all snake species names"""
        return list(self.knowledge_base.keys())
    
    def format_snake_info(self, snake_name):
        """Format snake information as display text"""
        info = self.get_snake_info(snake_name)
        if not info:
            return f"No detailed information available for {snake_name}.\n\nIf you have relevant information, welcome to provide it!"
        
        # Choose appropriate emoji based on safety level
        safety_level = "🚨" if "HIGHLY VENOMOUS" in info['safety'] else "✅"
        
        formatted_text = f"""
{safety_level} {snake_name} Detailed Profile

📖 Species Introduction
{info['description']}

🎯 Physical Characteristics
{info['characteristics']}

🏠 Habitat
{info['habitat']}

⚠️ Safety Assessment
{info['safety']}

🦎 Behavior and Habits
{info['behavior']}

🌍 Conservation Status
{info['conservation']}

💡 Practical Advice
{info['tips']}
        """
        return formatted_text.strip()
    
    def search_by_keyword(self, keyword):
        """Search for related snake species by keyword"""
        results = []
        keyword = keyword.lower()
        
        for snake_name, info in self.knowledge_base.items():
            # Search for keyword in all information fields
            all_text = " ".join(info.values()).lower()
            if keyword in all_text or keyword in snake_name.lower():
                results.append(snake_name)
        
        return results

# Create global knowledge base instance
snake_kb = SnakeKnowledge()

# For backward compatibility, provide a simple function interface
def get_snake_knowledge(snake_name):
    """Get snake knowledge information - simplified interface"""
    return snake_kb.format_snake_info(snake_name)

# If running this file directly, display all snake information
if __name__ == "__main__":
    print("🐍 Snake Knowledge Database Test")
    print("=" * 50)
    
    for snake_name in snake_kb.get_all_snakes():
        print(f"\n{snake_name}:")
        print("-" * 30)
        print(snake_kb.format_snake_info(snake_name))
        print("\n" + "="*50) 