db = db.getSiblingDB('gitquery');

db.createCollection('users', {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["user_id", "username", "created_at"],
            properties: {
                user_id: {
                    bsonType: "string",
                    description: "Unique user identifier"
                },
                username: {
                    bsonType: "string",
                    description: "Username"
                },
                email: {
                    bsonType: "string",
                    description: "User email"
                },
                preferences: {
                    bsonType: "object",
                    description: "User preferences"
                },
                github_token: {
                    bsonType: "string",
                    description: "GitHub API token (encrypted)"
                },
                created_at: {
                    bsonType: "date",
                    description: "Account creation timestamp"
                },
                last_login: {
                    bsonType: "date",
                    description: "Last login timestamp"
                }
            }
        }
    }
});

db.createCollection('chat_sessions', {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["session_id", "user_id", "created_at"],
            properties: {
                session_id: {
                    bsonType: "string",
                    description: "Unique session identifier"
                },
                user_id: {
                    bsonType: "string",
                    description: "User identifier"
                },
                title: {
                    bsonType: "string",
                    description: "Session title"
                },
                messages: {
                    bsonType: "array",
                    description: "Chat messages",
                    items: {
                        bsonType: "object",
                        required: ["role", "content", "timestamp"],
                        properties: {
                            role: {
                                bsonType: "string",
                                enum: ["user", "assistant", "system"]
                            },
                            content: {
                                bsonType: "string"
                            },
                            timestamp: {
                                bsonType: "date"
                            }
                        }
                    }
                },
                created_at: {
                    bsonType: "date",
                    description: "Session creation timestamp"
                },
                updated_at: {
                    bsonType: "date",
                    description: "Last update timestamp"
                }
            }
        }
    }
});

db.createCollection('user_interactions', {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["interaction_id", "user_id", "repo_id", "timestamp"],
            properties: {
                interaction_id: {
                    bsonType: "string",
                    description: "Unique interaction identifier"
                },
                user_id: {
                    bsonType: "string",
                    description: "User identifier"
                },
                repo_id: {
                    bsonType: "string",
                    description: "Repository identifier"
                },
                interaction_type: {
                    bsonType: "string",
                    enum: ["view", "star", "fork", "clone", "recommend"],
                    description: "Type of interaction"
                },
                timestamp: {
                    bsonType: "date",
                    description: "Interaction timestamp"
                },
                metadata: {
                    bsonType: "object",
                    description: "Additional interaction data"
                }
            }
        }
    }
});

db.createCollection('recommendations', {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["recommendation_id", "user_id", "repo_id", "created_at"],
            properties: {
                recommendation_id: {
                    bsonType: "string",
                    description: "Unique recommendation identifier"
                },
                user_id: {
                    bsonType: "string",
                    description: "User identifier"
                },
                repo_id: {
                    bsonType: "string",
                    description: "Recommended repository ID"
                },
                score: {
                    bsonType: "double",
                    description: "Recommendation score"
                },
                reason: {
                    bsonType: "string",
                    description: "Recommendation reason"
                },
                created_at: {
                    bsonType: "date",
                    description: "Creation timestamp"
                },
                feedback: {
                    bsonType: "object",
                    description: "User feedback"
                }
            }
        }
    }
});

db.users.createIndex({ "user_id": 1 }, { unique: true });
db.users.createIndex({ "username": 1 }, { unique: true });
db.users.createIndex({ "email": 1 }, { unique: true });
db.users.createIndex({ "created_at": -1 });

db.chat_sessions.createIndex({ "session_id": 1 }, { unique: true });
db.chat_sessions.createIndex({ "user_id": 1 });
db.chat_sessions.createIndex({ "created_at": -1 });
db.chat_sessions.createIndex({ "updated_at": -1 });

db.user_interactions.createIndex({ "interaction_id": 1 }, { unique: true });
db.user_interactions.createIndex({ "user_id": 1 });
db.user_interactions.createIndex({ "repo_id": 1 });
db.user_interactions.createIndex({ "timestamp": -1 });
db.user_interactions.createIndex({ "interaction_type": 1 });

db.recommendations.createIndex({ "recommendation_id": 1 }, { unique: true });
db.recommendations.createIndex({ "user_id": 1 });
db.recommendations.createIndex({ "repo_id": 1 });
db.recommendations.createIndex({ "created_at": -1 });
db.recommendations.createIndex({ "score": -1 });

print('✓ MongoDB initialized successfully');
