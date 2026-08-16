import uuid
from functools import lru_cache

from sqlalchemy import String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

DB_PATH = "data/application.sqlite"
DB_URI = f"sqlite:///{DB_PATH}"


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(20), unique=True)


class Book(Base):
    __tablename__ = "books"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(200), unique=True)


@lru_cache(maxsize=1)
def get_engine(db_uri: str):
    engine = create_engine(db_uri)
    Base.metadata.create_all(engine)
    return engine


@lru_cache(maxsize=1)
def get_session_factory(db_uri: str) -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(db_uri))


def get_session(db_uri: str) -> Session:
    session_factory = get_session_factory(db_uri)
    return session_factory()


if __name__ == "__main__":
    from rich import print

    user_names = ["hossam"]
    book_names = ["NLP with Transformers", "Fluent Python", "Intro to ML with Python"]

    session = get_session(DB_URI)

    for user_name in user_names:
        user = User(name=user_name)
        session.add(user)
        session.commit()
        session.refresh(user)
        print(f"Created user: '{user.name}'")

    # for book_name in book_names:
    #     book = Book(name=book_name)
    #     session.add(book)
    #     session.commit()
    #     session.refresh(book)
    #     print(f"Created book: '{book.name}'")

    session.close()
