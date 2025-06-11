def test_create_get_update_delete(instr_repo):
    # create
    inst = instr_repo.create(ticker="aapl", company_name="Apple Inc.")
    assert inst.id and inst.ticker == "AAPL"

    # get
    same = instr_repo.get_by_ticker("aapl")
    assert same.id == inst.id

    # update
    updated = instr_repo.update(inst.id, sector="TECH")
    assert updated.sector == "TECH"

    # duplicate create → той самий об’єкт
    dup = instr_repo.create(ticker="AAPL")
    assert dup.id == inst.id

    # delete
    rows = instr_repo.delete(inst.id)
    assert rows == 1
    assert instr_repo.get_by_id(inst.id) is None

