import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from src.gateway.services.user_service import UserService


@pytest.mark.asyncio
async def test_record_interaction_writes_canonical_event_with_mapped_action():
    user_interactions = SimpleNamespace(insert_one=AsyncMock())
    users = SimpleNamespace(update_one=AsyncMock())
    user_preferences = SimpleNamespace(update_one=AsyncMock())
    db = SimpleNamespace(
        user_interactions=user_interactions,
        users=users,
        user_preferences=user_preferences,
    )
    redis = SimpleNamespace(lpush=AsyncMock(), ltrim=AsyncMock())

    service = UserService(db, redis)

    await service.record_interaction(
        user_id="u1",
        repo_id="owner/repo",
        action="star",
        query="python fastapi",
        variant="baseline",
        position_in_results=2,
        metadata={"language": "Python"},
    )

    assert user_interactions.insert_one.await_count == 1
    event = user_interactions.insert_one.await_args.args[0]
    assert event["user_id"] == "u1"
    assert event["repo_id"] == "owner/repo"
    assert event["interaction_type"] == "save"
    assert event["variant"] == "baseline"
    assert event["position_in_results"] == 2


@pytest.mark.asyncio
async def test_get_interaction_history_prefers_canonical_collection():
    docs = [
        {"_id": "x1", "user_id": "u1", "repo_id": "a/b", "interaction_type": "click"},
        {"_id": "x2", "user_id": "u1", "repo_id": "c/d", "interaction_type": "view"},
    ]

    cursor = SimpleNamespace(to_list=AsyncMock(return_value=docs))
    limit_chain = MagicMock(return_value=cursor)
    sort_chain = MagicMock(return_value=SimpleNamespace(limit=limit_chain))
    user_interactions = SimpleNamespace(
        find=MagicMock(return_value=SimpleNamespace(sort=sort_chain))
    )
    users = SimpleNamespace(find_one=AsyncMock(return_value=None))
    db = SimpleNamespace(
        user_interactions=user_interactions,
        users=users,
        user_preferences=SimpleNamespace(),
    )
    redis = SimpleNamespace(
        get=AsyncMock(return_value=None),
        setex=AsyncMock(),
        lpush=AsyncMock(),
        ltrim=AsyncMock(),
    )

    service = UserService(db, redis)

    history = await service.get_interaction_history("u1", limit=2)

    assert len(history) == 2
    assert "_id" not in history[0]
    assert history[0]["interaction_type"] == "click"


@pytest.mark.asyncio
async def test_update_preferences_syncs_recommender_collection():
    users = SimpleNamespace(update_one=AsyncMock())
    user_preferences = SimpleNamespace(
        find_one=AsyncMock(return_value={"total_interactions": 7}),
        update_one=AsyncMock(),
    )
    db = SimpleNamespace(users=users, user_preferences=user_preferences)
    redis = SimpleNamespace(delete=AsyncMock())

    service = UserService(db, redis)

    updated = await service.update_preferences(
        "u1",
        {
            "languages": ["Python", "TypeScript"],
            "topics": ["ai", "search"],
        },
    )

    assert updated.languages == ["Python", "TypeScript"]
    assert user_preferences.update_one.await_count == 1
    payload = user_preferences.update_one.await_args.args[1]
    assert payload["$set"]["language_preferences"]["Python"] == 1.0
    assert payload["$set"]["topic_preferences"]["ai"] == 1.0


@pytest.mark.asyncio
async def test_replace_user_chats_rewrites_collection():
    chats_collection = SimpleNamespace(delete_many=AsyncMock(), insert_many=AsyncMock())
    db = SimpleNamespace(get_collection=MagicMock(return_value=chats_collection))
    redis = SimpleNamespace()
    service = UserService(db, redis)

    chats = [
        {
            "id": "chat-1",
            "title": "Session",
            "timestamp": 1710000000000,
            "messages": [
                {"content": "hello", "role": "user", "timestamp": 1710000000000}
            ],
        }
    ]

    result = await service.replace_user_chats("u1", chats)

    assert db.get_collection.call_args.args[0] == "user_chats"
    assert chats_collection.delete_many.await_args.args[0] == {"user_id": "u1"}
    assert chats_collection.insert_many.await_count == 1
    assert result[0]["id"] == "chat-1"
    assert "sort_order" not in result[0]


@pytest.mark.asyncio
async def test_get_saved_repos_reads_ordered_docs():
    docs = [
        {
            "repo_id": "owner/repo",
            "name": "repo",
            "owner": "owner",
            "description": "desc",
            "stars": 5,
            "forks": 1,
            "language": "Python",
            "url": "https://github.com/owner/repo",
        }
    ]
    cursor = SimpleNamespace(to_list=AsyncMock(return_value=docs))
    limit_chain = MagicMock(return_value=cursor)
    sort_chain = MagicMock(return_value=SimpleNamespace(limit=limit_chain))
    saved_collection = SimpleNamespace(
        find=MagicMock(return_value=SimpleNamespace(sort=sort_chain))
    )

    db = SimpleNamespace(get_collection=MagicMock(return_value=saved_collection))
    redis = SimpleNamespace()
    service = UserService(db, redis)

    repos = await service.get_saved_repos("u1", limit=10)

    assert db.get_collection.call_args.args[0] == "user_saved_repos"
    assert repos == [
        {
            "id": "owner/repo",
            "name": "repo",
            "owner": "owner",
            "description": "desc",
            "stars": 5,
            "forks": 1,
            "language": "Python",
            "url": "https://github.com/owner/repo",
        }
    ]


@pytest.mark.asyncio
async def test_replace_user_folders_rewrites_collection():
    folders_collection = SimpleNamespace(
        delete_many=AsyncMock(), insert_many=AsyncMock()
    )
    db = SimpleNamespace(get_collection=MagicMock(return_value=folders_collection))
    redis = SimpleNamespace()
    service = UserService(db, redis)

    folders = [
        {
            "id": "folder-1",
            "name": "Pinned",
            "items": [{"id": "owner/repo", "name": "repo"}],
            "expanded": True,
        }
    ]

    result = await service.replace_user_folders("u1", folders)

    assert db.get_collection.call_args.args[0] == "user_folders"
    assert folders_collection.delete_many.await_args.args[0] == {"user_id": "u1"}
    assert folders_collection.insert_many.await_count == 1
    assert result[0]["id"] == "folder-1"
    assert result[0]["expanded"] is True
    assert "sort_order" not in result[0]


@pytest.mark.asyncio
async def test_get_user_folders_reads_ordered_docs():
    docs = [
        {
            "folder_id": "folder-1",
            "name": "Pinned",
            "items": [{"id": "owner/repo", "name": "repo"}],
            "expanded": False,
        }
    ]
    cursor = SimpleNamespace(to_list=AsyncMock(return_value=docs))
    limit_chain = MagicMock(return_value=cursor)
    sort_chain = MagicMock(return_value=SimpleNamespace(limit=limit_chain))
    folders_collection = SimpleNamespace(
        find=MagicMock(return_value=SimpleNamespace(sort=sort_chain))
    )

    db = SimpleNamespace(get_collection=MagicMock(return_value=folders_collection))
    redis = SimpleNamespace()
    service = UserService(db, redis)

    folders = await service.get_user_folders("u1", limit=10)

    assert db.get_collection.call_args.args[0] == "user_folders"
    assert folders == [
        {
            "id": "folder-1",
            "name": "Pinned",
            "items": [{"id": "owner/repo", "name": "repo"}],
            "expanded": False,
        }
    ]
