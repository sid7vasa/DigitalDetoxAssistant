import wikipedia as wiki


def extract_info(skill_set):
    skill_summary_list = []
    li = []
    for i, skill in enumerate(skill_set):
        skill = skill.lower()
        print(skill)
        try:
            skill_summary = wiki.WikipediaPage(skill).summary
            skill_summary_list.append(skill_summary)
        except Exception as e:
            li.append(skill)
            skill_summary_list.append("")
            print(e)
        print("=" * 100)
