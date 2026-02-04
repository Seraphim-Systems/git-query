// Azure Cosmos DB Initialization Script
// Note: For production, use actual Azure Cosmos DB service
// This is a MongoDB API emulation for local development

db = db.getSiblingDB('gitquery_cosmos');

// Create collections for large-scale data

// GitHub repositories metadata
db.createCollection('repositories', {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["repo_id", "full_name", "owner", "created_at"],
            properties: {
                repo_id: {
                    bsonType: "string",
                    description: "GitHub repository ID"
                },
                full_name: {
                    bsonType: "string",
                    description: "Full repository name (owner/repo)"
                },
                owner: {
                    bsonType: "string",
                    description: "Repository owner"
                },
                name: {
                    bsonType: "string",
                    description: "Repository name"
                },
                description: {
                    bsonType: "string",
                    description: "Repository description"
                },
                stars: {
                    bsonType: "int",
                    description: "Star count"
                },
                forks: {
                    bsonType: "int",
                    description: "Fork count"
                },
                language: {
                    bsonType: "string",
                    description: "Primary language"
                },
                topics: {
                    bsonType: "array",
                    description: "Repository topics"
                },
                created_at: {
                    bsonType: "date",
                    description: "Creation timestamp"
                },
                updated_at: {
                    bsonType: "date",
                    description: "Last update timestamp"
                }
            }
        }
    }
});

// Repository activity logs (commits, releases, etc.)
db.createCollection('repository_activity', {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["repo_id", "activity_type", "timestamp"],
            properties: {
                repo_id: {
                    bsonType: "string",
                    description: "Repository ID"
                },
                activity_type: {
                    bsonType: "string",
                    enum: ["commit", "release", "issue", "pull_request"],
                    description: "Type of activity"
                },
                timestamp: {
                    bsonType: "date",
                    description: "Activity timestamp"
                },
                details: {
                    bsonType: "object",
                    description: "Activity details"
                }
            }
        }
    }
});

// Create indexes for efficient queries
db.repositories.createIndex({ "repo_id": 1 }, { unique: true });
db.repositories.createIndex({ "full_name": 1 }, { unique: true });
db.repositories.createIndex({ "owner": 1 });
db.repositories.createIndex({ "language": 1 });
db.repositories.createIndex({ "stars": -1 });
db.repositories.createIndex({ "updated_at": -1 });
db.repositories.createIndex({ "topics": 1 });

db.repository_activity.createIndex({ "repo_id": 1, "timestamp": -1 });
db.repository_activity.createIndex({ "activity_type": 1 });

print("Azure Cosmos DB (MongoDB API) initialization completed");
