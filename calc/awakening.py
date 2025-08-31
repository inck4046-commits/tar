AWK = {
    "(진각)체":   {"hp":144,"atk":12,"def":12},
    "(진각)공":   {"hp":72, "atk":30,"def":12},
    "(진각)방":   {"hp":96, "atk":24,"def":12},
    "(진각)체공": {"hp":96, "atk":18,"def":18},
    "(진각)체방": {"hp":96, "atk":12,"def":24},
    "(진각)공방": {"hp":72, "atk":24,"def":18},
}
def get_awakening_stat(type_name:str)->dict:
    return AWK.get(type_name, {"hp":0,"atk":0,"def":0})
