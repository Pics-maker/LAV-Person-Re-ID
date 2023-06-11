def data_set_mapper(data_set: str, angle=None):
    """
    构建CASIA-B数据集对应的ID-nm-angle映射
    :param data_set: 构建的数据集是什么(train, verify, test)
    :return: ['001_nm-01_000', '001_nm-01_018', '001_nm-01_036', ...]
    """
    if data_set == "train":
        ID = [1, 84]
        NM = [1, 2, 3, 4]
    if data_set == "verify":
        ID = [1, 84]
        NM = [5, 6]
    if data_set == "test":
        ID = [85, 124]
        NM = [1, 2, 5, 6]

    mapper = []
    for id in range(ID[0], ID[1] + 1):
        for nm in NM:
            if angle is None:  # 注意：ID 109的 nm-{01~04} 中的"126", "144", "162", "180"没有数据
                if id == 109 and (nm == 1 or nm == 2 or nm == 3 or nm == 4):
                    angle_list = ["000", "018", "036", "054", "072", "090", "108"]
                else:
                    angle_list = ["000", "018", "036", "054", "072", "090", "108", "126", "144", "162", "180"]
            else:
                angle_list = angle

            for a in angle_list:
                mapper.append(str(id).zfill(3) + "_nm-" + str(nm).zfill(2) + "_" + a)

    return mapper
