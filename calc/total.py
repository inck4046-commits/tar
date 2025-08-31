import pandas as pd

def calculate_stat_total(
    type_stat: dict,
    awakening_stat: dict,
    gem_stat: dict,
    potion_stat: dict,
    accessory_percent: dict,
    enchant_percent: dict,
    spirit_percent: dict,
    spirit_flat: dict,
    collection_stat: dict,
    spirit_addition_stat: dict,
    buff_percent: dict,
    penalty_stat: dict = None
) -> dict:
    """
    Excel 공식:
    =INT(INT((C3+C4)*(1+C5+C6))*(1+C7)+(C8*(1+C7)))+C9+C10+C11-C12
    """
    if penalty_stat is None:
        penalty_stat = {'hp': 0, 'atk': 0, 'def': 0}

    C3 = {k: type_stat[k] + awakening_stat[k] + gem_stat[k] for k in ('hp', 'atk', 'def')}
    C4 = potion_stat
    C5 = accessory_percent
    C6 = enchant_percent
    C7 = spirit_percent
    C8 = spirit_flat
    C9 = collection_stat
    C10 = spirit_addition_stat
    C11 = buff_percent
    C12 = penalty_stat

    final = {}
    for stat in ('hp', 'atk', 'def'):
        base = C3[stat] + C4[stat]
        mult1 = 1 + C5.get(stat, 0) + C6.get(stat, 0)
        step1 = int(base * mult1)
        step2 = int(step1 * (1 + C7.get(stat, 0)) + C8.get(stat, 0) * (1 + C7.get(stat, 0)))
        total = step2 + C9.get(stat, 0) + C10.get(stat, 0) + C11.get(stat, 0) - C12.get(stat, 0)
        final[stat] = total

    bibel = final['hp'] * final['atk'] * final['def']
    ivel = int((final['hp'] // 4 + final['atk'] + final['def']) * 4)

    return {
        'HP': final['hp'],
        'ATK': final['atk'],
        'DEF': final['def'],
        '비벨': bibel,
        '이벨': ivel
    }

if __name__ == '__main__':
    # 테스트 입력 예시
    type_stat = {'hp': 1113, 'atk': 156, 'def': 156}
    awakening_stat = {'hp': 96, 'atk': 12, 'def': 24}
    gem_stat = {'hp': 0, 'atk': 166, 'def': 46}
    potion_stat = {'hp': 24, 'atk': 6, 'def': 6}
    accessory_percent = {'hp': 0.0, 'atk': 0.19, 'def': 0.0}
    enchant_percent = {'hp': 0.0, 'atk': 0.21, 'def': 0.0}
    spirit_percent = {'hp': 0.0, 'atk': 0.28, 'def': 0.0}
    spirit_flat = {'hp': 0, 'atk': 0, 'def': 0}
    collection_stat = {'hp': 0, 'atk': 54, 'def': 0}
    spirit_addition_stat = {'hp': 0, 'atk': 0, 'def': 0}
    buff_percent = {'hp': 0, 'atk': 0, 'def': 0}
    penalty_stat = {'hp': 0, 'atk': 0, 'def': 0}

    result = calculate_stat_total(
        type_stat, awakening_stat, gem_stat,
        potion_stat, accessory_percent, enchant_percent,
        spirit_percent, spirit_flat,
        collection_stat, spirit_addition_stat,
        buff_percent, penalty_stat
    )
    print(result)

