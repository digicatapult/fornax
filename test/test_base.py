import unittest
from sqlalchemy import create_engine
from sqlalchemy.orm.session import Session
import fornax.model as model

Base = model.Base


class TestCaseDB(unittest.TestCase):
    """ A base test for setting up and tearing down the databse """

    @classmethod
    def setUpClass(cls):
        """ Create the engine, create one connection and start a transaction """
        engine = create_engine(
            'postgresql://postgres:fornax@localhost:5433/postgres', 
            echo=False
        )
        connection = engine.connect()
        cls._engine = engine
        cls._connection = connection
        cls.__transaction = connection.begin()
        Base.metadata.create_all(connection)

    @classmethod
    def tearDownClass(cls):
        """ tear down the top level transaction """
        cls.__transaction.rollback()
        cls._connection.close()
        cls._engine.dispose()

    def setUp(self):
        """ create a new session and a nested transaction """
        self._transaction = self._connection.begin_nested()
        self.session = Session(self._connection)

    def tearDown(self):
        """ rollback the nested transaction """
        self._transaction.rollback()
        self.session.close()