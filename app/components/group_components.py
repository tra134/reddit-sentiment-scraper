import streamlit as st
from sqlalchemy.orm import Session
from app.services.user_service import UserService
from app.services.recommendation_service import RecommendationService
from app.services.trend_service import TrendService

def show_group_management(db: Session):
    """Display group management interface"""
    st.header("📊 Manage Your Interest Groups")
    
    user_service = UserService(db)
    current_user = st.session_state.user
    
    # Add new group
    with st.expander("➕ Add New Group", expanded=True):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            group_name = st.text_input("Group Name", placeholder="e.g., AI News, Tech Updates")
        
        with col2:
            subreddit = st.text_input("Subreddit", placeholder="e.g., artificial, technology")
        
        if st.button("Add Group"):
            if group_name and subreddit:
                try:
                    user_service.add_user_group(
                        user_id=current_user["id"],
                        group_name=group_name,
                        subreddit=subreddit
                    )
                    st.success(f"Group '{group_name}' added successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding group: {e}")
            else:
                st.error("Please fill in all fields")
    
    # Display current groups
    st.subheader("Your Groups")
    user_groups = user_service.get_user_groups(current_user["id"])
    
    if not user_groups:
        st.info("You haven't added any groups yet. Add some groups to get started!")
        return
    
    for group in user_groups:
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.write(f"**{group.group_name}**")
            st.caption(f"r/{group.subreddit}")
        
        with col2:
            if st.button("📊 Trends", key=f"trends_{group.id}"):
                st.session_state.current_group = group.subreddit
        
        with col3:
            if st.button("🗑️ Remove", key=f"remove_{group.id}"):
                if user_service.remove_user_group(current_user["id"], group.id):
                    st.success("Group removed successfully!")
                    st.rerun()

def show_recommendations(db: Session):
    """Display personalized recommendations"""
    st.header("🎯 Personalized Recommendations")
    
    current_user = st.session_state.user
    trend_service = TrendService()
    recommendation_service = RecommendationService(db, trend_service)
    
    # Get recommendations
    recommendations = recommendation_service.get_group_recommendations(current_user["id"])
    
    if not recommendations:
        st.info("No recommendations available. Add some groups first!")
        return
    
    for i, rec in enumerate(recommendations):
        with st.container():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader(f"#{i+1} {rec['trend_topic']}")
                st.write(f"**Group:** {rec['group_name']} (r/{rec['subreddit']})")
                st.write(f"**Posts analyzed:** {rec['post_count']}")
            
            with col2:
                sentiment_color = "green" if rec['sentiment_score'] > 0 else "red"
                st.metric(
                    "Sentiment Score",
                    f"{rec['sentiment_score']:.2f}",
                    delta_color="off"
                )
                st.metric(
                    "Trend Score",
                    f"{rec['trend_score']:.2f}",
                    delta_color="normal"
                )
            
            st.markdown("---")

def show_suggested_groups(db: Session):
    """Display suggested groups to follow"""
    st.header("💡 Suggested Groups")
    
    current_user = st.session_state.user
    recommendation_service = RecommendationService(db, TrendService())
    user_service = UserService(db)
    
    suggestions = recommendation_service.get_suggested_groups(current_user["id"])
    
    for suggestion in suggestions:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"**{suggestion['name']}**")
            st.caption(f"r/{suggestion['subreddit']}")
        
        with col2:
            if st.button("➕ Follow", key=f"follow_{suggestion['subreddit']}"):
                user_service.add_user_group(
                    user_id=current_user["id"],
                    group_name=suggestion['name'],
                    subreddit=suggestion['subreddit']
                )
                st.success(f"Now following {suggestion['name']}!")
                st.rerun()