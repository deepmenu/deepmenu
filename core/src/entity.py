class EntityBase:
  def __init__(self, id):
    self.id = id

  def to_dict(self):
    return self.__dict__


class OrderEntry(EntityBase):
  pass


class OrderItem(EntityBase):
  pass


class MenuItem(EntityBase):
  def __init__(self, id, name, description, category=None):
    super(MenuItem, self).__init__(id)
    self.name = name
    self.description = description
    self.category = category


class ShiftItem():
  pass