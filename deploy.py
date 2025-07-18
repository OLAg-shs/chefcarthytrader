import os
from app import app, db
from models import User
from werkzeug.security import generate_password_hash
from dotenv import load_dotenv

def init_db():
    with app.app_context():
        # Create database tables
        db.create_all()
        
        # Create admin user if not exists
        admin = User.query.filter_by(username=os.getenv("ADMIN_USER")).first()
        if not admin:
            admin = User(
                username=os.getenv("ADMIN_USER"),
                password=generate_password_hash(os.getenv("ADMIN_PWD")),
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()
            print("✅ Admin user created successfully!")
        else:
            print("ℹ️ Admin user already exists.")

if __name__ == "__main__":
    load_dotenv()
    
    print("🔄 Initializing database...")
    init_db()
    print("✅ Database initialized successfully!")
    
    print("\n🚀 Starting the application...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)