db = db.getSiblingDB('gitquery');
db.createCollection('users');
db.createCollection('chat_sessions');
db.createCollection('user_interactions');
db.createCollection('recommendations');
db.users.createIndex({ 'user_id': 1 }, { unique: true });
db.users.createIndex({ 'username': 1 }, { unique: true });
db.users.createIndex({ 'email': 1 }, { unique: true });
db.chat_sessions.createIndex({ 'session_id': 1 }, { unique: true });
db.chat_sessions.createIndex({ 'user_id': 1 });
db.user_interactions.createIndex({ 'interaction_id': 1 }, { unique: true });
db.user_interactions.createIndex({ 'user_id': 1 });
db.user_interactions.createIndex({ 'repo_id': 1 });
db.recommendations.createIndex({ 'recommendation_id': 1 }, { unique: true });
db.recommendations.createIndex({ 'user_id': 1 });
db.recommendations.createIndex({ 'repo_id': 1 });

// Seed default admin user (password: admin123 - CHANGE IN PRODUCTION)
const adminExists = db.users.findOne({ email: 'admin@gitquery.local' });
if (!adminExists) {
    db.users.insertOne({
        user_id: 'admin@gitquery.local',
        email: 'admin@gitquery.local',
        username: 'admin',
        password_hash: '240be518fabd2724ddb6f04eeb1da5967448d7e831c08c8fa822809f74c720a9',
        created_at: new Date(),
        is_admin: true
    });
    print('MongoDB: admin user seeded (change default password!)');
} else {
    print('MongoDB: admin user already exists, skipping seed');
}

// Seed the env-configured admin account (WEB_ADMIN_EMAIL / WEB_ADMIN_USERNAME)
// injected via docker-compose environment at init time.
const envAdminEmail = process.env.WEB_ADMIN_EMAIL;
const envAdminUsername = process.env.WEB_ADMIN_USERNAME;
const envAdminPassword = process.env.WEB_ADMIN_PASSWORD;
if (envAdminEmail && envAdminPassword && envAdminEmail !== 'admin@gitquery.local') {
    const envAdminExists = db.users.findOne({ email: envAdminEmail });
    if (!envAdminExists) {
        // SHA-256 of the plain password — gateway uses same scheme for now
        const crypto = require('crypto');
        const hash = crypto.createHash('sha256').update(envAdminPassword).digest('hex');
        db.users.insertOne({
            user_id: envAdminEmail,
            email: envAdminEmail,
            username: envAdminUsername || envAdminEmail.split('@')[0],
            password_hash: hash,
            created_at: new Date(),
            is_admin: true
        });
        print('MongoDB: env admin user seeded (' + envAdminEmail + ')');
    } else {
        print('MongoDB: env admin user already exists, skipping seed');
    }
}

print(' MongoDB initialized');
